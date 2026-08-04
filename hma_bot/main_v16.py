"""HMA Simple Sentinel production runtime.

Startup is standalone so Telegram receives one authoritative message only.
Railway shutdown is graceful: stop the processing loop first, then close the
OKX client once the active symbol operation has returned.
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
        self._shutdown_requested = False
        self._client_closed = False

    def request_shutdown(self) -> None:
        """Ask the loops to finish without closing OKX underneath active work."""
        if not self._shutdown_requested:
            _LOG.info("Graceful shutdown requested; finishing active symbol work")
        self._shutdown_requested = True
        self._running = False

    async def stop(self):
        """Close OKX exactly once, after run_forever has stopped using it."""
        self.request_shutdown()
        if self._client_closed:
            return
        self._client_closed = True
        try:
            await self.client.close()
        except Exception as exc:
            text = str(exc).lower()
            if "closed by the user" not in text and "already closed" not in text:
                _LOG.warning("OKX close during shutdown failed: %s", exc)
        _LOG.info("HMA Simple Sentinel shutdown complete")

    async def run_forever(self):
        """Process symbols without reporting expected redeploy shutdown as errors."""
        while self._running:
            for symbol in self.cfg.symbols:
                if not self._running:
                    break
                try:
                    await self._process(symbol)
                except asyncio.CancelledError:
                    self.request_shutdown()
                    raise
                except Exception as exc:
                    if self._shutdown_requested or not self._running:
                        _LOG.info(
                            "[%s] active operation ended during graceful shutdown: %s",
                            symbol,
                            exc,
                        )
                        break

                    _LOG.error("[%s] unhandled: %s", symbol, exc, exc_info=True)
                    now = time.time()
                    last = self._error_notified_at.get(symbol, 0.0)
                    if now - last >= self.ERROR_NOTIFY_COOLDOWN_SEC:
                        self._error_notified_at[symbol] = now
                        try:
                            await self.tg.send_text(
                                f"❌ `{v15.v14.v13.v12.v11.v10.v9.v8.v7.v5.v4.v3.base._sym(symbol)}` "
                                f"error: {str(exc)[:150]}\n"
                                "Telegram repeats muted for 15 minutes; Railway log has traceback."
                            )
                        except Exception:
                            pass

            if not self._running:
                break
            self._maybe_status_log()
            try:
                await asyncio.sleep(self.cfg.poll_interval_sec)
            except asyncio.CancelledError:
                self.request_shutdown()
                raise

    def _set_view_v3(self, symbol: str, df5, df15, df1h, df4h):
        try:
            if self.open_position_count() >= self.cfg.max_positions:
                self._view[symbol] = f"POSITION LIMIT | MAX {self.cfg.max_positions}"
                return
            px = float(df5["close"].iloc[-1]) if len(df5) else 0.0
            self._view[symbol] = (
                f"5M px={px:.6g} | "
                f"{self.strat.entry_status(df4h, df1h, df15, df5)}"
            )
        except Exception as exc:
            self._view[symbol] = f"view error: {str(exc)[:140]}"

    async def start(self):
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start: " + "; ".join(problems))

        if not self.cfg.paper:
            if not await self.client.ensure_hedge_mode():
                raise RuntimeError("Could not confirm OKX hedge mode.")

        balance = await self.client.fetch_balance_usdt()
        _LOG.info(
            "=== HMA SIMPLE SENTINEL [%s] symbols=%s margin=$%.2f "
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
                f"⚡ *HMA Simple Sentinel — {mode}*\n"
                f"Symbols: `{', '.join(self.cfg.symbols)}`\n"
                f"Balance: `{balance:.2f}` USDT | Margin `${self.cfg.margin_per_position_usd:.2f}`/position "
                f"| Leverage `x{self.cfg.leverage}` | Max `{self.cfg.max_positions}` positions\n\n"
                "Simplified entry logic:\n"
                f"`Layer 1 · 1H Direction` Trend `≥{self.strat.one_h_early_min:.0f}` · edge `≥{self.strat.one_h_direction_edge_min:.0f}` · Q `≥{self.strat.quality_min:.0f}`\n"
                f"`Layer 2 · 15M Setup` S/R within `{self.strat.setup_proximity_atr:.2f} ATR` or aligned EMA20 pullback\n"
                f"`Layer 3 · 5M Trigger` EMA8/13, BOS/CHOCH or continuation within `{self.strat.exec_trigger_lookback}` closed bars\n"
                f"`Risk Check` Room `≥{self.strat.min_room_atr:.2f} ATR` and actual R:R `≥{self.strat.min_actual_rr:.2f}`\n"
                "4H and Confidence are diagnostic only; neither can block an entry.\n"
                "Risk management: Stage 1 `+0.7%→lock +0.4%` | Stage 2 `+1.1%→lock +0.75%` | Final TP `+1.5%`\n"
                "Schedule: `FX 24/5 new entries` | Existing positions managed `24/7`\n"
                "Recovery: `OKX-native SL/TP synchronised after restart`"
            )

        _LOG.info(
            "HMA Simple Sentinel startup complete: 3 layers + 1 risk check; "
            "status and order creation use the same decision"
        )


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(
                getattr(
                    v15.v14.v13.v12.v11.v10.v9.v8.v7.v5.v4.v3.base._signal,
                    sig_name,
                ),
                bot.request_shutdown,
            )
        except (NotImplementedError, AttributeError):
            pass

    await bot.start()
    try:
        await bot.run_forever()
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(_main())
