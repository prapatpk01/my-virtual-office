"""Trend Pullback Continuation (TPC Sentinel) production runtime.

Continuous 1H quality:
    ADX 45 + CHOP 35 + directional DMI 20.

Quality policy:
    Q >= 60   normal S1/S2 or R1/R2 hold/reclaim
    Q 45..59  S2/R2 hold/reclaim, or reclaim at S1/R1
    Q < 45    no trade

The TPC trading logic is identical for every symbol. Asset profiles adjust
only execution-zone width and the structure-stop ATR floor/cap.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import replace

import numpy as np

import main_v15 as v15
import strategy_v12 as S
from sentinel_context import build_context, trend_score_4h

_LOG = v15._LOG


class Bot(v15.Bot):
    ASSET_PROFILES = {
        "DEFAULT": {"zone_atr5": 0.25, "sl_min_atr15": 1.00, "sl_max_atr15": 1.80},
        "BTC": {"zone_atr5": 0.22, "sl_min_atr15": 1.00, "sl_max_atr15": 1.60},
        "ETH": {"zone_atr5": 0.25, "sl_min_atr15": 1.00, "sl_max_atr15": 1.60},
        "SOL": {"zone_atr5": 0.30, "sl_min_atr15": 1.10, "sl_max_atr15": 1.70},
        "HYPE": {"zone_atr5": 0.35, "sl_min_atr15": 1.20, "sl_max_atr15": 1.80},
        "XRP": {"zone_atr5": 0.30, "sl_min_atr15": 1.10, "sl_max_atr15": 1.70},
        "TRX": {"zone_atr5": 0.25, "sl_min_atr15": 1.00, "sl_max_atr15": 1.60},
        "XAU": {"zone_atr5": 0.25, "sl_min_atr15": 1.20, "sl_max_atr15": 1.80},
        "CL": {"zone_atr5": 0.30, "sl_min_atr15": 1.30, "sl_max_atr15": 2.00},
    }

    def __init__(self):
        super().__init__()
        self.strat = S.PrecisionTrendStructureV12(self.cfg.strategy_config())
        self._shutdown_requested = False
        self._client_closed = False
        self._risk_symbol = ""
        self._active_profile_name = "DEFAULT"
        self._active_profile = dict(self.ASSET_PROFILES["DEFAULT"])

        self._quality_base = self.strat.quality_state_1h
        self.strat.quality_conditional_min = 45.0
        self.strat.quality_full_min = 60.0
        self.strat.quality_min = 45.0

        def continuous_quality(df1h):
            clean = self._clean_quality_frame(df1h)
            base = self._quality_base(clean)
            values = (base.adx, base.chop, base.plus_di, base.minus_di)
            if clean is None or len(clean) < 60 or not all(
                math.isfinite(float(v)) for v in values
            ):
                return type(base)(0.0, 0.0, 100.0, 0.0, 0.0)

            long_score = float(trend_score_4h(clean, "long"))
            short_score = float(trend_score_4h(clean, "short"))
            dmi_edge = (
                float(base.plus_di) - float(base.minus_di)
                if long_score >= short_score
                else float(base.minus_di) - float(base.plus_di)
            )
            adx_points = self._adx_points(base.adx)
            chop_points = self._chop_points(base.chop)
            dmi_points = self._dmi_points(dmi_edge)
            q = float(np.clip(adx_points + chop_points + dmi_points, 0.0, 100.0))
            return type(base)(
                q,
                float(base.adx),
                float(base.chop),
                float(base.plus_di),
                float(base.minus_di),
            )

        self.strat.quality_state_1h = continuous_quality

        original_evaluate = self.strat.evaluate

        def tiered_evaluate(df4h, df1h, df15, df5):
            decision = original_evaluate(df4h, df1h, df15, df5)
            quality = decision.quality
            if quality is None or not (45.0 <= float(quality.q) < 60.0):
                return decision
            if decision.context is None:
                return decision

            level = str(decision.context.location.zone or "")
            trigger = decision.execution[0] if decision.execution else ""
            deep_level = level in {"S2", "R2"}
            reclaim = "RECLAIM" in trigger
            if deep_level or reclaim:
                return decision

            stage = "L3_TRIGGER" if level in {"S1", "R1"} else "L2_SETUP"
            blocker = (
                f"Q {quality.q:.1f} CONDITIONAL: {level or 'S/R'} hold disabled; "
                "need S2/R2 or closed-5M reclaim"
            )
            return replace(
                decision,
                ready=False,
                stage=stage,
                blocker=blocker,
                execution=None,
            )

        self.strat.evaluate = tiered_evaluate

        original_risk_plan = self.strat._risk_plan

        def adaptive_risk_plan(decision, df15, df5):
            plan = original_risk_plan(decision, df15, df5)
            if plan is None:
                return None
            entry, sl, tp, atr15, structure_level, rr = plan
            if atr15 > 0:
                profile = self._active_profile
                min_atr = float(profile["sl_min_atr15"])
                max_atr = float(profile["sl_max_atr15"])
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
                rr = abs(float(tp) - float(entry)) / max(stop_distance, 1e-12)
            return entry, sl, tp, atr15, structure_level, rr

        self.strat._risk_plan = adaptive_risk_plan

    @staticmethod
    def _base_symbol(symbol: str) -> str:
        text = str(symbol or "").upper().strip()
        for separator in ("/", "-", ":"):
            if separator in text:
                text = text.split(separator, 1)[0]
        return text

    def _apply_asset_profile(self, symbol: str):
        name = self._base_symbol(symbol)
        profile = self.ASSET_PROFILES.get(name, self.ASSET_PROFILES["DEFAULT"])
        self._risk_symbol = symbol
        self._active_profile_name = name if name in self.ASSET_PROFILES else "DEFAULT"
        self._active_profile = dict(profile)
        self.strat.sr_touch_zone_atr5 = float(profile["zone_atr5"])
        return profile

    def _asset_profile_status(self) -> str:
        p = self._active_profile
        return (
            f"Profile={self._active_profile_name} "
            f"Zone={p['zone_atr5']:.2f}ATR5 "
            f"SL={p['sl_min_atr15']:.2f}-{p['sl_max_atr15']:.2f}ATR15"
        )

    @staticmethod
    def _clean_quality_frame(df1h):
        if df1h is None or len(df1h) == 0:
            return df1h
        clean = df1h.copy()
        clean = clean[~clean.index.duplicated(keep="last")].sort_index()
        required = [c for c in ("open", "high", "low", "close") if c in clean]
        if required:
            clean[required] = clean[required].replace([np.inf, -np.inf], np.nan)
            clean = clean.dropna(subset=required)
        return clean

    @staticmethod
    def _adx_points(adx: float) -> float:
        return float(np.clip((float(adx) - 8.0) / 22.0 * 45.0, 0.0, 45.0))

    @staticmethod
    def _chop_points(chop: float) -> float:
        return float(np.clip((70.0 - float(chop)) / 25.0 * 35.0, 0.0, 35.0))

    @staticmethod
    def _dmi_points(edge: float) -> float:
        return float(np.clip((float(edge) + 5.0) / 20.0 * 20.0, 0.0, 20.0))

    @staticmethod
    def _quality_tier(q: float) -> str:
        if q >= 60.0:
            return "FULL"
        if q >= 45.0:
            return "CONDITIONAL"
        return "BLOCK"

    def _quality_status(self, df1h) -> str:
        try:
            clean = self._clean_quality_frame(df1h)
            if clean is None or len(clean) < 60:
                return f"QData=WARMUP({0 if clean is None else len(clean)})"
            quality = self.strat.quality_state_1h(clean)
            values = (
                quality.q,
                quality.adx,
                quality.chop,
                quality.plus_di,
                quality.minus_di,
            )
            if not all(math.isfinite(float(v)) for v in values):
                return "QData=INVALID_BLOCKED"

            long_score = float(trend_score_4h(clean, "long"))
            short_score = float(trend_score_4h(clean, "short"))
            edge = (
                float(quality.plus_di) - float(quality.minus_di)
                if long_score >= short_score
                else float(quality.minus_di) - float(quality.plus_di)
            )
            return (
                f"Q={quality.q:.1f}/{self._quality_tier(quality.q)} "
                f"ADX={quality.adx:.1f}({self._adx_points(quality.adx):.1f}/45) "
                f"CHOP={quality.chop:.1f}({self._chop_points(quality.chop):.1f}/35) "
                f"DMIedge={edge:+.1f}({self._dmi_points(edge):.1f}/20)"
            )
        except Exception as exc:
            _LOG.debug("Quality diagnostics unavailable: %s", exc)
            return "QData=ERROR_BLOCKED"

    def request_shutdown(self) -> None:
        if not self._shutdown_requested:
            _LOG.info("Graceful shutdown requested; finishing active symbol work")
        self._shutdown_requested = True
        self._running = False

    async def stop(self):
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
        _LOG.info("TPC Sentinel shutdown complete")

    async def run_forever(self):
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
                        _LOG.info("[%s] operation ended during shutdown: %s", symbol, exc)
                        break
                    _LOG.error("[%s] unhandled: %s", symbol, exc, exc_info=True)
                    now = time.time()
                    last = self._error_notified_at.get(symbol, 0.0)
                    if now - last >= self.ERROR_NOTIFY_COOLDOWN_SEC:
                        self._error_notified_at[symbol] = now
                        try:
                            sym = v15.v14.v13.v12.v11.v10.v9.v8.v7.v5.v4.v3.base._sym(symbol)
                            await self.tg.send_text(
                                f"❌ `{sym}` error: {str(exc)[:150]}\n"
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
        return f"{float(value):.6g}"

    def _entry_zone_status(self, df5, df15, df1h, df4h) -> str:
        try:
            if len(df5) < 20 or len(df1h) < 60 or len(df15) < 90:
                return ""
            direction, quality = self.strat._simple_direction(df1h)
            if direction.side is None or quality.q < 45.0:
                return ""
            context = build_context(
                df15=df15,
                df1h=df1h,
                df4h=df4h,
                side="long" if direction.side == S.Side.LONG else "short",
            )
            _, name, level_price, _, _ = self.strat._sr_entry_state(
                df5, context.location, direction.side
            )
            if level_price is None or not math.isfinite(float(level_price)):
                return ""
            d5 = df5.copy()
            d5["atr"] = self.strat._atr(d5, self.strat.cfg.atr_len)
            atr5 = float(d5["atr"].iloc[-1])
            if not math.isfinite(atr5) or atr5 <= 0:
                return ""
            level = float(level_price)
            half = max(self.strat.sr_touch_zone_atr5 * atr5, abs(level) * 1e-6)
            low, high = level - half, level + half
            price = float(df5["close"].iloc[-1])
            if low <= price <= high:
                distance = "IN_ZONE"
            elif price > high:
                distance = f"{(price - high) / atr5:.2f}ATR_ABOVE"
            else:
                distance = f"{(low - price) / atr5:.2f}ATR_BELOW"
            return (
                f"EntryZone={name} {self._fmt_zone_price(low)}-"
                f"{self._fmt_zone_price(high)} | ZoneDist={distance}"
            )
        except Exception as exc:
            _LOG.debug("Entry-zone display unavailable: %s", exc)
            return ""

    def _set_view_v3(self, symbol: str, df5, df15, df1h, df4h):
        try:
            self._apply_asset_profile(symbol)
            if self.open_position_count() >= self.cfg.max_positions:
                self._view[symbol] = f"POSITION LIMIT | MAX {self.cfg.max_positions}"
                return
            px = float(df5["close"].iloc[-1]) if len(df5) else 0.0
            zone = self._entry_zone_status(df5, df15, df1h, df4h)
            q = self._quality_status(df1h)
            status = self.strat.entry_status(df4h, df1h, df15, df5)
            zone_part = f" | {zone}" if zone else ""
            self._view[symbol] = (
                f"5M px={px:.6g} | {self._asset_profile_status()}"
                f"{zone_part} | {q} | {status}"
            )
        except Exception as exc:
            self._view[symbol] = f"view error: {str(exc)[:140]}"

    async def _look_for_entry(self, symbol: str, st: dict):
        self._apply_asset_profile(symbol)
        return await super()._look_for_entry(symbol, st)

    async def start(self):
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start: " + "; ".join(problems))
        if not self.cfg.paper and not await self.client.ensure_hedge_mode():
            raise RuntimeError("Could not confirm OKX hedge mode.")

        balance = await self.client.fetch_balance_usdt()
        _LOG.info(
            "=== TPC SENTINEL V1.0 [%s] symbols=%s margin=$%.2f leverage=x%d max_pos=%d balance=%.2f ===",
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
                f"🎯 *TPC Sentinel v1.0 — Trend Pullback Continuation — {mode}*\n"
                f"Symbols: `{', '.join(self.cfg.symbols)}`\n"
                f"Balance: `{balance:.2f}` USDT | Margin `${self.cfg.margin_per_position_usd:.2f}`/position "
                f"| Leverage `x{self.cfg.leverage}` | Max `{self.cfg.max_positions}` positions\n\n"
                "Quality: `ADX 45 + CHOP 35 + directional DMI 20`\n"
                "`Q ≥60` normal S1/S2 or R1/R2 hold/reclaim\n"
                "`Q 45–59` only S2/R2, or reclaim at S1/R1\n"
                "`Q <45` no trade\n"
                "LONG uses adaptive 15M `S1/S2`; SHORT uses `R1/R2`\n"
                "Hold: touch without closing through, then next closed 5M candle confirms\n"
                "Reclaim: close through, then first closed 5M candle reclaims the level\n"
                "Asset profiles: execution-zone width + ATR stop only; entry logic is unchanged\n"
                "Prepared: `BTC ETH SOL HYPE XRP TRX XAU CL`\n"
                "Stage 1 `+0.7%→lock +0.4%` | Stage 2 `+1.1%→lock +0.75%` | Final TP `+1.5%`\n"
                "4H and Confidence are diagnostic only."
            )

        _LOG.info(
            "TPC Sentinel v1.0 startup complete: continuous Q 45/35/20; "
            "multi-asset execution/risk profiles active"
        )


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            signal_module = v15.v14.v13.v12.v11.v10.v9.v8.v7.v5.v4.v3.base._signal
            loop.add_signal_handler(getattr(signal_module, sig_name), bot.request_shutdown)
        except (NotImplementedError, AttributeError):
            pass
    await bot.start()
    try:
        await bot.run_forever()
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(_main())
