"""HMA S/R Sentinel production runtime.

Startup is standalone so Telegram receives one authoritative message only.
Railway shutdown is graceful: stop the processing loop first, then close the
OKX client once the active symbol operation has returned.
"""
from __future__ import annotations

import asyncio
import math
import time

import numpy as np

import main_v15 as v15
import strategy_v12 as S
from sentinel_context import build_context

_LOG = v15._LOG


class Bot(v15.Bot):
    def __init__(self):
        super().__init__()
        self.strat = S.PrecisionTrendStructureV12(self.cfg.strategy_config())
        self._shutdown_requested = False
        self._client_closed = False
        self._risk_symbol = ""

        # Keep the original calculator for diagnostics, but route all strategy
        # decisions through a safe wrapper.  NaN/inf quality values previously
        # made comparisons such as ``q < minimum`` evaluate False, which could
        # allow invalid market data to bypass the quality gate.
        self._quality_original = self.strat.quality_state_1h

        def safe_quality_state(df1h):
            clean = self._clean_quality_frame(df1h)
            quality = self._quality_original(clean)
            values = (
                quality.q,
                quality.adx,
                quality.chop,
                quality.plus_di,
                quality.minus_di,
            )
            if len(clean) < 60 or not all(math.isfinite(float(v)) for v in values):
                return type(quality)(0.0, 0.0, 100.0, 0.0, 0.0)
            return quality

        self.strat.quality_state_1h = safe_quality_state

        # XAU needs more breathing room than crypto on 5M execution. Keep the
        # structure-selected direction, but enforce a 15M ATR floor/cap for the
        # actual stop used by status, risk validation and live order creation.
        original_risk_plan = self.strat._risk_plan

        def adaptive_risk_plan(decision, df15, df5):
            plan = original_risk_plan(decision, df15, df5)
            if plan is None:
                return None

            entry, sl, tp, atr15, structure_level, rr = plan
            symbol = str(self._risk_symbol).upper()
            if symbol.startswith("XAU") and atr15 > 0:
                min_atr = 1.20
                max_atr = 1.80
                raw_distance = abs(float(entry) - float(sl))
                stop_distance = max(
                    min_atr * atr15,
                    min(raw_distance, max_atr * atr15),
                )
                sl = (
                    float(entry) - stop_distance
                    if decision.side == S.Side.LONG
                    else float(entry) + stop_distance
                )
                rr = abs(float(tp) - float(entry)) / max(
                    stop_distance, 1e-12
                )

            return entry, sl, tp, atr15, structure_level, rr

        self.strat._risk_plan = adaptive_risk_plan

    @staticmethod
    def _clean_quality_frame(df1h):
        """Return sorted, unique, finite closed-1H OHLC rows for Q."""
        if df1h is None or len(df1h) == 0:
            return df1h
        clean = df1h.copy()
        clean = clean[~clean.index.duplicated(keep="last")].sort_index()
        required = [column for column in ("open", "high", "low", "close") if column in clean]
        if required:
            clean[required] = clean[required].replace([np.inf, -np.inf], np.nan)
            clean = clean.dropna(subset=required)
        return clean

    def _quality_status(self, df1h) -> str:
        """Expose the exact inputs behind Q and distinguish Q=0 from bad data."""
        try:
            clean = self._clean_quality_frame(df1h)
            if clean is None or len(clean) < 60:
                count = 0 if clean is None else len(clean)
                return f"QData=WARMUP({count})"

            quality = self._quality_original(clean)
            values = (
                quality.q,
                quality.adx,
                quality.chop,
                quality.plus_di,
                quality.minus_di,
            )
            if not all(math.isfinite(float(v)) for v in values):
                return "QData=INVALID_BLOCKED"

            adx_points = self.strat._adx_score(float(quality.adx))
            chop_points = self.strat._chop_score(float(quality.chop))
            if quality.q <= 0.05:
                if quality.adx <= 10.0 and quality.chop >= 62.0:
                    state = "Q0_VALID_LOW_ADX_HIGH_CHOP"
                else:
                    state = "Q0_CHECK"
            elif quality.q < self.strat.quality_min:
                state = "Q_BELOW_MIN"
            else:
                state = "Q_PASS"

            return (
                f"QRaw={quality.q:.1f} "
                f"ADX={quality.adx:.1f}({adx_points:.1f}) "
                f"CHOP={quality.chop:.1f}({chop_points:.1f}) "
                f"DI+={quality.plus_di:.1f} DI-={quality.minus_di:.1f} "
                f"QState={state}"
            )
        except Exception as exc:
            _LOG.debug("Quality diagnostics unavailable: %s", exc)
            return "QData=ERROR_BLOCKED"

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
        _LOG.info("HMA S/R Sentinel shutdown complete")

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

    @staticmethod
    def _fmt_zone_price(value: float) -> str:
        """Compact price formatting that still preserves crypto precision."""
        return f"{float(value):.6g}"

    def _entry_zone_status(self, df5, df15, df1h, df4h) -> str:
        """Return the exact active S1/S2 or R1/R2 price band for logs.

        The displayed band is the same 5M ATR touch zone used by the entry
        engine, so Railway shows where price must trade before hold/reclaim
        confirmation can create an order.
        """
        try:
            if len(df5) < 20 or len(df1h) < 60 or len(df15) < 90:
                return ""

            direction, quality = self.strat._simple_direction(df1h)
            if direction.side is None or quality.q < self.strat.quality_min:
                return ""

            context = build_context(
                df15=df15,
                df1h=df1h,
                df4h=df4h,
                side="long" if direction.side == S.Side.LONG else "short",
            )
            _, level_name, level_price, _, _ = self.strat._sr_entry_state(
                df5, context.location, direction.side
            )
            if level_price is None or not math.isfinite(float(level_price)):
                return ""

            d5 = df5.copy()
            d5["atr"] = self.strat._atr(d5, self.strat.cfg.atr_len)
            atr5 = float(d5["atr"].iloc[-1])
            if not math.isfinite(atr5) or atr5 <= 0.0:
                return ""

            level = float(level_price)
            half_width = max(
                self.strat.sr_touch_zone_atr5 * atr5,
                abs(level) * 1e-6,
            )
            zone_low = level - half_width
            zone_high = level + half_width
            price = float(df5["close"].iloc[-1])

            if zone_low <= price <= zone_high:
                distance_text = "IN_ZONE"
            elif price > zone_high:
                distance_text = f"{(price - zone_high) / atr5:.2f}ATR_ABOVE"
            else:
                distance_text = f"{(zone_low - price) / atr5:.2f}ATR_BELOW"

            return (
                f"EntryZone={level_name} "
                f"{self._fmt_zone_price(zone_low)}-"
                f"{self._fmt_zone_price(zone_high)} | "
                f"ZoneDist={distance_text}"
            )
        except Exception as exc:
            _LOG.debug("Entry-zone display unavailable: %s", exc)
            return ""

    def _set_view_v3(self, symbol: str, df5, df15, df1h, df4h):
        try:
            self._risk_symbol = symbol
            if self.open_position_count() >= self.cfg.max_positions:
                self._view[symbol] = f"POSITION LIMIT | MAX {self.cfg.max_positions}"
                return

            px = float(df5["close"].iloc[-1]) if len(df5) else 0.0
            zone_status = self._entry_zone_status(df5, df15, df1h, df4h)
            quality_status = self._quality_status(df1h)
            strategy_status = self.strat.entry_status(df4h, df1h, df15, df5)
            zone_part = f" | {zone_status}" if zone_status else ""
            self._view[symbol] = (
                f"5M px={px:.6g}{zone_part} | {quality_status} | "
                f"{strategy_status}"
            )
        except Exception as exc:
            self._view[symbol] = f"view error: {str(exc)[:140]}"

    async def _look_for_entry(self, symbol: str, st: dict):
        self._risk_symbol = symbol
        return await super()._look_for_entry(symbol, st)

    async def start(self):
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start: " + "; ".join(problems))

        if not self.cfg.paper:
            if not await self.client.ensure_hedge_mode():
                raise RuntimeError("Could not confirm OKX hedge mode.")

        balance = await self.client.fetch_balance_usdt()
        _LOG.info(
            "=== HMA S/R SENTINEL [%s] symbols=%s margin=$%.2f "
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
                f"🎯 *HMA S/R Sentinel — {mode}*\n"
                f"Symbols: `{', '.join(self.cfg.symbols)}`\n"
                f"Balance: `{balance:.2f}` USDT | Margin `${self.cfg.margin_per_position_usd:.2f}`/position "
                f"| Leverage `x{self.cfg.leverage}` | Max `{self.cfg.max_positions}` positions\n\n"
                "Entry logic:\n"
                f"`Layer 1 · 1H Direction` Trend `≥{self.strat.one_h_early_min:.0f}` · edge `≥{self.strat.one_h_direction_edge_min:.0f}` · Q `≥{self.strat.quality_min:.0f}`\n"
                "`LONG Levels` adaptive 15M `S1/S2`\n"
                "`SHORT Levels` adaptive 15M `R1/R2`\n"
                "`Hold Entry` touch a level without closing through it, then enter after the next closed 5M candle still holds the level\n"
                "`Reclaim Entry` close through a level, then enter immediately on the first closed 5M candle reclaiming back through it\n"
                "`Log Zone` shows the active S/R entry price band and distance from current price\n"
                "`Q Diagnostics` shows raw ADX/CHOP/DI values; invalid data is blocked\n"
                f"`Risk Check` Room `≥{self.strat.min_room_atr:.2f} ATR` and actual R:R `≥{self.strat.min_actual_rr:.2f}`\n"
                "`XAU Stop` 15M structure with `1.20–1.80 ATR` distance\n"
                "TP/SL and position management remain unchanged.\n"
                "4H and Confidence are diagnostic only; neither can block an entry.\n"
                "Risk management: Stage 1 `+0.7%→lock +0.4%` | Stage 2 `+1.1%→lock +0.75%` | Final TP `+1.5%`\n"
                "Schedule: `FX 24/5 new entries` | Existing positions managed `24/7`\n"
                "Recovery: `OKX-native SL/TP synchronised after restart`"
            )

        _LOG.info(
            "HMA S/R Sentinel startup complete: 1H direction plus S1/S2 or "
            "R1/R2 hold/reclaim entries; logs show Q inputs and active entry "
            "zone; invalid Q data is blocked; XAU structure stop uses "
            "1.20-1.80 ATR; status and order creation use the same decision"
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
