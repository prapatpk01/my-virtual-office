"""EMA Hybrid A+B Quality V2.1 runtime wrapper.

Keeps the proven EMA Hybrid runtime/journal/Telegram lifecycle code in main_core.py
and adds quality-v2 startup diagnostics plus the XAU/XAG same-direction exposure guard.
Also normalizes legacy client helpers that may be synchronous even though the EMA
runtime awaits them, preventing NoneType/tuple await crashes during position management.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import signal

import main_core as core

_LOG = core._LOG


class Bot(core.Bot):
    METAL_CORR_GUARD = os.getenv("EMA_METAL_CORR_GUARD", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }

    def __init__(self):
        super().__init__()
        self._install_client_await_compat()
        self.strat.correlation_guard = self._metal_correlation_blocked

    def _install_client_await_compat(self) -> None:
        """Wrap legacy sync helpers so awaited EMA lifecycle code is safe."""
        quantize = getattr(self.client, "quantize_amount", None)
        if callable(quantize) and not inspect.iscoroutinefunction(quantize):
            async def _quantize_amount(*args, __fn=quantize, **kwargs):
                result = __fn(*args, **kwargs)
                if result is None:
                    _LOG.warning("[EMA COMPAT] quantize_amount returned None; using zero-sized result")
                    return 0.0, 0.0
                return result
            self.client.quantize_amount = _quantize_amount
            _LOG.info("[EMA COMPAT] wrapped sync quantize_amount as awaitable")

        move_sl = getattr(self.client, "move_sl_to_breakeven", None)
        if callable(move_sl) and not inspect.iscoroutinefunction(move_sl):
            async def _move_sl_to_breakeven(*args, __fn=move_sl, **kwargs):
                result = __fn(*args, **kwargs)
                return bool(result) if result is not None else False
            self.client.move_sl_to_breakeven = _move_sl_to_breakeven
            _LOG.info("[EMA COMPAT] wrapped sync move_sl_to_breakeven as awaitable")

    async def _entry_frames(self, symbol: str):
        frames = await super()._entry_frames(symbol)
        for frame in frames:
            try:
                frame.attrs["symbol"] = symbol
            except Exception:
                pass
        return frames

    def _metal_correlation_blocked(self, symbol: str, side) -> bool:
        """Do not add XAU and XAG in the same direction at the same time."""
        if not self.METAL_CORR_GUARD:
            return False

        root = str(symbol).split("/")[0].upper()
        if root not in {"XAU", "XAG"}:
            return False

        peer_root = "XAG" if root == "XAU" else "XAU"
        peer_symbol = next(
            (s for s in self.cfg.symbols if str(s).split("/")[0].upper() == peer_root),
            None,
        )
        if not peer_symbol:
            return False

        peer_pos = (self.state.get(peer_symbol) or {}).get("pos") or {}
        peer_side = str(peer_pos.get("side") or "").lower()
        wanted = str(getattr(side, "value", side) or "").lower()
        blocked = peer_side in {"long", "short"} and peer_side == wanted

        if blocked:
            self._view[symbol] = (
                f"CORR BLOCK | {root} {wanted.upper()} blocked because "
                f"{peer_root} {peer_side.upper()} is already open"
            )
            _LOG.info(
                "[%s] correlation block: %s %s already open",
                symbol, peer_root, peer_side.upper(),
            )
        return blocked

    async def start(self):
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start: " + "; ".join(problems))
        if not self.cfg.paper and not await self.client.ensure_hedge_mode():
            raise RuntimeError("Could not confirm OKX hedge mode.")

        balance = await self.client.fetch_balance_usdt()
        _LOG.info(
            "=== EMA HYBRID A+B QUALITY V2.1 [%s] symbols=%s margin=$%.2f "
            "leverage=x%d max_pos=%d balance=%.2f ===",
            "PAPER" if self.cfg.paper else "LIVE",
            self.cfg.symbols,
            self.cfg.margin_per_position_usd,
            self.cfg.leverage,
            self.cfg.max_positions,
            balance,
        )

        await self._reconcile_startup()
        self._running = True

        if self.tg.enabled:
            asyncio.create_task(self._command_loop())
            mode = "PAPER" if self.cfg.paper else "LIVE"
            await self.tg.send_text(
                f"📈 *EMA Hybrid A+B Quality V2.1 — {mode}*\n"
                f"Symbols: `{', '.join(self.cfg.symbols)}`\n"
                f"Balance: `{balance:.2f}` USDT | Margin `${self.cfg.margin_per_position_usd:.2f}`/position "
                f"| Leverage `x{self.cfg.leverage}` | Max `{self.cfg.max_positions}` positions\n\n"
                f"15M Bias: LONG Close>SMA{self.strat.SMA_LEN} + RSI≥`{self.strat.BIAS_RSI_LONG_MIN:.0f}` "
                f"+ SMA slope UP | SHORT Close<SMA{self.strat.SMA_LEN} + RSI≤`{self.strat.BIAS_RSI_SHORT_MAX:.0f}` "
                "+ SMA slope DOWN\n"
                f"A: EMA{self.strat.EMA_FAST}/{self.strat.EMA_SLOW} fresh cross + direction candle + expanding spread\n"
                f"B1 Reclaim: EMA13 ±`{self.strat.PULLBACK_TOUCH_ATR:.2f} ATR` true-zone + EMA13 slope + ADX `≥{self.strat.ADX_MIN:.0f}` + CHOP `≤{self.strat.CHOP_MAX:.0f}`\n"
                f"B2 Micro BOS: break `≥{self.strat.MICRO_BOS_BREAK_ATR:.2f} ATR` beyond structure + EMA spread expanding + ADX `≥{self.strat.MICRO_BOS_ADX_MIN:.0f}` rising + CHOP `≤{self.strat.MICRO_BOS_CHOP_MAX:.0f}`\n"
                f"SL Gate: `{self.strat.SL_MIN_PCT*100:.2f}%–{self.strat.SL_MAX_PCT*100:.2f}%` "
                f"| Structure buffer `{self.strat.SL_BUFFER_ATR:.2f} ATR`\n"
                f"TP1: `+{self.TP1_R:.1f}R` → trim `{self.TP1_TRIM_PCT*100:.0f}%` → SL `BE+{self.TP1_LOCK_R:.2f}R`\n"
                f"TP2: next 5M liquidity/swing with room `≥{self.strat.TP2_MIN_RR:.1f}R`\n"
                f"XAU/XAG same-direction guard: `{'ON' if self.METAL_CORR_GUARD else 'OFF'}`\n"
                "Telegram: Entry + Setup Engine + TP1 + TP2/SL/TP1_LOCK alerts\n"
                "PAPER entries: `24/7` | LIVE entries: `24/5` | Open positions managed: `24/7`"
            )

        _LOG.info(
            "EMA Hybrid A+B Quality V2.1 startup complete: strict Micro BOS, "
            "15M bias, SL sanity, metal correlation guard and await-compat active"
        )


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
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
