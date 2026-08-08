"""EMA Hybrid Pro runtime using the existing OKX execution infrastructure."""
from __future__ import annotations

import asyncio
import os
import signal
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HMA = os.path.join(ROOT, "hma_bot")
if HMA not in sys.path:
    sys.path.insert(0, HMA)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import main_v15 as base
from strategy import EMAHybridProStrategy

_LOG = base._LOG


class Bot(base.Bot):
    def __init__(self):
        super().__init__()
        self.strat = EMAHybridProStrategy(self.cfg.strategy_config())
        self._okx_reconnect_lock = asyncio.Lock()

    async def _recover_okx_session(self, reason: str = "") -> None:
        """Re-open CCXT's async HTTP session after an unexpected close.

        Railway/container networking can occasionally leave aiohttp's session
        closed while the process itself is still healthy.  CCXT then raises
        `okx instance was closed by the user` on every subsequent request.
        Re-opening the existing exchange object is safe and preserves its
        markets, credentials and rate limiter.
        """
        async with self._okx_reconnect_lock:
            exchange = getattr(self.client, "_exchange", None)
            if exchange is None:
                return
            session = getattr(exchange, "session", None)
            if session is not None and not getattr(session, "closed", False):
                return
            _LOG.warning("[OKX] async session closed; reopening%s", f" ({reason})" if reason else "")
            await exchange.open()
            _LOG.info("[OKX] async session reopened")

    @staticmethod
    def _is_closed_session_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "instance was closed by the user" in text
            or "session is closed" in text
            or "client session is closed" in text
        )

    async def _frame(self, symbol, tf, minutes, limit=300):
        """Fetch candles with one automatic OKX session recovery retry."""
        now_ms = int(__import__("time").time() * 1000)
        try:
            raw = await self.client.fetch_ohlcv(symbol, tf, limit=limit)
        except Exception as exc:
            if not self._is_closed_session_error(exc):
                raise
            await self._recover_okx_session(f"{symbol} {tf}")
            raw = await self.client.fetch_ohlcv(symbol, tf, limit=limit)

        df = base._ohlcv_to_df(raw)
        if df.empty:
            return df
        close_ms = (df.index.as_unit("ns").asi8 // 1_000_000) + minutes * 60_000
        return df[close_ms <= now_ms]

    def _set_view_v3(self, symbol: str, df5, df15, df1h, df4h):
        try:
            if self.open_position_count() >= self.cfg.max_positions:
                self._view[symbol] = f"EMA HYBRID POSITION LIMIT | MAX {self.cfg.max_positions}"
                return
            px = float(df15["close"].iloc[-1]) if len(df15) else 0.0
            self._view[symbol] = (
                f"15M px={px:.6g} | {self.strat.entry_status(df4h, df1h, df15, df5)}"
            )
        except Exception as exc:
            self._view[symbol] = f"EMA Hybrid view error: {str(exc)[:140]}"

    async def start(self):
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start: " + "; ".join(problems))
        if not self.cfg.paper and not await self.client.ensure_hedge_mode():
            raise RuntimeError("Could not confirm OKX hedge mode.")

        balance = await self.client.fetch_balance_usdt()
        _LOG.info(
            "=== EMA HYBRID PRO [%s] symbols=%s margin=$%.2f leverage=x%d max_pos=%d balance=%.2f ===",
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
                f"📈 *EMA Hybrid Pro — {mode}*\n"
                f"Symbols: `{', '.join(self.cfg.symbols)}`\n"
                f"Balance: `{balance:.2f}` USDT | Margin `${self.cfg.margin_per_position_usd:.2f}`/position "
                f"| Leverage `x{self.cfg.leverage}` | Max `{self.cfg.max_positions}` positions\n\n"
                "Primary TF: `M15` | Trend confirm: `H1`\n"
                "Trend gate: `EMA20 > EMA50 > EMA200` for LONG; reverse for SHORT on both H1 + M15\n"
                "Location: `Fib 50%-61.8%` + touch `EMA20/EMA50`\n"
                "Trigger: `Liquidity Sweep + closed-M15 Price Action`\n"
                "PA: `Engulfing / Pin Bar / Inside Break / Break & Retest`\n"
                "Volume: soft confirmation when available\n"
                "SL: `beyond swing + 0.15 ATR`\n"
                "TP1 milestone: `2R` → protect profit | Final TP: `3R`\n"
                "Minimum initial RR: `1:2` | No chase entry."
            )
        _LOG.info("EMA Hybrid Pro startup complete")


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
