"""Trend Pullback Continuation (TPC Sentinel) production runtime.

Continuous 1H quality:
    ADX 45 + CHOP 35 + directional DMI 20.

Quality policy:
    Q >= 60   normal S1/S2 or R1/R2 hold/reclaim
    Q 45..59  S2/R2 hold/reclaim, or reclaim at S1/R1
    Q < 45    no trade

The TPC trading logic is identical for every symbol. Asset profiles adjust
only execution-zone width and the structure-stop ATR floor/cap.

CTR is an isolated fallback engine. TPC is always evaluated first; CTR may
only enter when TPC has no valid signal for that symbol and no position exists.
CTR also requires enough open distance to the pending primary TPC zone, and
its target must finish before that zone.
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
        "XAG": {"zone_atr5": 0.30, "sl_min_atr15": 1.20, "sl_max_atr15": 1.90},
        "CL": {"zone_atr5": 0.30, "sl_min_atr15": 1.30, "sl_max_atr15": 2.00},
    }

    CTR_MARGIN_MULTIPLIER = 0.40
    CTR_MIN_Q = 55.0
    CTR_MIN_TREND = 60.0
    CTR_MAX_STOP_ATR15 = 1.00
    CTR_MIN_RR = 0.80
    CTR_MAX_TP_PCT = 0.007
    CTR_MIN_TPC_GAP_PCT = 0.009
    CTR_MIN_TPC_GAP_ATR15 = 1.20
    CTR_TPC_ZONE_BUFFER_ATR15 = 0.10

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

        # Keep the primary TPC generator untouched. CTR is called only when
        # TPC returns no signal, so the original entry logic remains dominant.
        self._tpc_generate_entry = self.strat.generate_entry

        def combined_generate_entry(
            df4h, df1h, df15, df5, has_open_position: bool = False
        ):
            tpc_signal = self._tpc_generate_entry(
                df4h, df1h, df15, df5, has_open_position=has_open_position
            )
            if tpc_signal is not None or has_open_position:
                return tpc_signal
            return self._ctr_generate_entry(df4h, df1h, df15, df5)

        self.strat.generate_entry = combined_generate_entry

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

    @staticmethod
    def _wma(values, length: int):
        weights = np.arange(1, length + 1, dtype=float)
        return values.rolling(length).apply(
            lambda x: float(np.dot(x, weights) / weights.sum()), raw=True
        )

    @classmethod
    def _hma16(cls, close):
        half = cls._wma(close, 8)
        full = cls._wma(close, 16)
        return cls._wma(2.0 * half - full, 4)

    @staticmethod
    def _rsi14(close) -> float:
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        value = float(rsi.iloc[-1])
        return value if math.isfinite(value) else 50.0

    def _tpc_gap_plan(self, df4h, df1h, df15, primary, entry: float, atr15: float):
        """Measure free counter-trend travel before the nearest primary TPC zone."""
        if primary.side is None or entry <= 0.0 or atr15 <= 0.0:
            return None

        primary_context = build_context(
            df15=df15,
            df1h=df1h,
            df4h=df4h,
            side="long" if primary.side == S.Side.LONG else "short",
        )
        levels = self.strat._side_levels(primary_context.location, primary.side)
        if not levels:
            return None

        zone_half = max(
            float(self._active_profile["zone_atr5"]) * atr15,
            abs(entry) * 1e-6,
        )
        buffer = self.CTR_TPC_ZONE_BUFFER_ATR15 * atr15

        if primary.side == S.Side.LONG:
            valid = [(name, level) for name, level in levels if float(level) < entry]
            if not valid:
                return None
            name, level = max(valid, key=lambda item: float(item[1]))
            zone_edge = float(level) + zone_half
            gap = entry - zone_edge
            target_guard = zone_edge + buffer
            available_reward = entry - target_guard
        else:
            valid = [(name, level) for name, level in levels if float(level) > entry]
            if not valid:
                return None
            name, level = min(valid, key=lambda item: float(item[1]))
            zone_edge = float(level) - zone_half
            gap = zone_edge - entry
            target_guard = zone_edge - buffer
            available_reward = target_guard - entry

        if gap <= 0.0 or available_reward <= 0.0:
            return None

        gap_pct = gap / max(entry, 1e-12)
        gap_atr = gap / max(atr15, 1e-12)
        eligible = (
            gap_pct >= self.CTR_MIN_TPC_GAP_PCT
            or gap_atr >= self.CTR_MIN_TPC_GAP_ATR15
        )
        return {
            "name": str(name),
            "level": float(level),
            "zone_edge": float(zone_edge),
            "target_guard": float(target_guard),
            "gap": float(gap),
            "gap_pct": float(gap_pct),
            "gap_atr": float(gap_atr),
            "available_reward": float(available_reward),
            "eligible": bool(eligible),
        }

    def _ctr_gap_status(self, df4h, df1h, df15) -> str:
        try:
            if len(df1h) < 60 or len(df15) < 90:
                return "TPCGap=WARMUP"
            primary, _ = self.strat._simple_direction(df1h)
            if primary.side is None:
                return "TPCGap=NO_DIRECTION"
            d15 = df15.copy()
            d15["atr"] = self.strat._atr(d15, self.strat.cfg.atr_len)
            atr15 = float(d15["atr"].iloc[-1])
            entry = float(d15["close"].iloc[-1])
            plan = self._tpc_gap_plan(df4h, df1h, d15, primary, entry, atr15)
            if plan is None:
                return "TPCGap=UNAVAILABLE"
            state = "OK" if plan["eligible"] else "SMALL"
            return (
                f"TPCGap={plan['name']} {plan['gap_pct'] * 100:.2f}%/"
                f"{plan['gap_atr']:.2f}ATR {state}"
            )
        except Exception as exc:
            _LOG.debug("TPC-gap diagnostics unavailable: %s", exc)
            return "TPCGap=ERROR"

    def _ctr_generate_entry(self, df4h, df1h, df15, df5):
        """Create a small, short-horizon counter-trend signal.

        TPC remains primary. CTR needs an established 1H trend, opposing S/R,
        HMA16 reversal, rejection, exhaustion and sufficient distance to the
        pending primary TPC zone. Its TP is capped before that zone.
        """
        if len(df1h) < 60 or len(df15) < 90 or len(df5) < 20:
            return None

        primary, quality = self.strat._simple_direction(df1h)
        if (
            primary.side is None
            or primary.score < self.CTR_MIN_TREND
            or quality.q < self.CTR_MIN_Q
        ):
            return None

        counter_side = (
            S.Side.SHORT if primary.side == S.Side.LONG else S.Side.LONG
        )
        context = build_context(
            df15=df15,
            df1h=df1h,
            df4h=df4h,
            side="short" if counter_side == S.Side.SHORT else "long",
        )
        levels = self.strat._side_levels(context.location, counter_side)
        if not levels:
            return None

        d15 = df15.copy()
        d15["atr"] = self.strat._atr(d15, self.strat.cfg.atr_len)
        close = d15["close"].astype(float)
        atr15 = float(d15["atr"].iloc[-1])
        if not math.isfinite(atr15) or atr15 <= 0.0:
            return None

        hma = self._hma16(close)
        ema20 = close.ewm(span=20, adjust=False).mean()
        basis = close.rolling(20).mean()
        std = close.rolling(20).std(ddof=0)
        upper = basis + 2.0 * std
        lower = basis - 2.0 * std
        values = [
            hma.iloc[-1],
            hma.iloc[-2],
            ema20.iloc[-1],
            upper.iloc[-1],
            lower.iloc[-1],
        ]
        if not all(math.isfinite(float(v)) for v in values):
            return None

        current = d15.iloc[-1]
        previous = d15.iloc[-2]
        entry = float(current["close"])
        tpc_gap = self._tpc_gap_plan(df4h, df1h, d15, primary, entry, atr15)
        if tpc_gap is None or not tpc_gap["eligible"]:
            return None

        zone = max(
            float(self._active_profile["zone_atr5"]) * atr15,
            abs(entry) * 1e-6,
        )

        candidates = []
        for name, level in levels:
            if counter_side == S.Side.SHORT:
                touched = (
                    max(float(current["high"]), float(previous["high"]))
                    >= level - zone
                )
                valid_side = entry <= level + zone
            else:
                touched = (
                    min(float(current["low"]), float(previous["low"]))
                    <= level + zone
                )
                valid_side = entry >= level - zone
            if touched and valid_side:
                candidates.append((name, float(level), abs(entry - float(level))))
        if not candidates:
            return None
        level_name, level, _ = min(candidates, key=lambda item: item[2])

        body = abs(float(current["close"]) - float(current["open"]))
        candle_range = max(
            float(current["high"]) - float(current["low"]), 1e-12
        )
        upper_wick = float(current["high"]) - max(
            float(current["open"]), float(current["close"])
        )
        lower_wick = min(
            float(current["open"]), float(current["close"])
        ) - float(current["low"])

        recent_high = float(d15["high"].iloc[-3:].max())
        recent_low = float(d15["low"].iloc[-3:].min())
        rsi = self._rsi14(close)

        if counter_side == S.Side.SHORT:
            hma_flip = (
                entry < float(hma.iloc[-1])
                and float(hma.iloc[-1]) < float(hma.iloc[-2])
            )
            rejection = (
                float(current["close"]) < float(current["open"])
                or upper_wick >= max(body, 0.30 * candle_range)
            )
            exhaustion = [
                (recent_high - float(ema20.iloc[-1])) / atr15 >= 1.20,
                (recent_high - float(hma.iloc[-1])) / atr15 >= 0.80,
                rsi >= 68.0,
                recent_high >= float(upper.iloc[-1]),
            ]
            if not hma_flip or not rejection:
                return None
            required = 3 if quality.q >= 85.0 else 2
            if sum(bool(x) for x in exhaustion) < required:
                return None
            raw_sl = (
                max(level, float(d15["high"].iloc[-5:].max()))
                + 0.15 * atr15
            )
            risk = raw_sl - entry
            if risk <= 0.0 or risk > self.CTR_MAX_STOP_ATR15 * atr15:
                return None
            risk = max(risk, 0.35 * atr15)
            sl = entry + risk
            reward = min(
                risk,
                entry * self.CTR_MAX_TP_PCT,
                float(tpc_gap["available_reward"]),
            )
            ema_reward = entry - float(ema20.iloc[-1])
            if 0.80 * risk <= ema_reward <= reward:
                reward = ema_reward
            rr = reward / max(risk, 1e-12)
            if reward <= 0.0 or rr < self.CTR_MIN_RR:
                return None
            tp = entry - reward
            if tp <= float(tpc_gap["target_guard"]):
                return None
            trigger = f"CTR_{level_name}_HMA16_REJECTION_SHORT"
            compat_trend = S.Trend.BULL
        else:
            hma_flip = (
                entry > float(hma.iloc[-1])
                and float(hma.iloc[-1]) > float(hma.iloc[-2])
            )
            rejection = (
                float(current["close"]) > float(current["open"])
                or lower_wick >= max(body, 0.30 * candle_range)
            )
            exhaustion = [
                (float(ema20.iloc[-1]) - recent_low) / atr15 >= 1.20,
                (float(hma.iloc[-1]) - recent_low) / atr15 >= 0.80,
                rsi <= 32.0,
                recent_low <= float(lower.iloc[-1]),
            ]
            if not hma_flip or not rejection:
                return None
            required = 3 if quality.q >= 85.0 else 2
            if sum(bool(x) for x in exhaustion) < required:
                return None
            raw_sl = (
                min(level, float(d15["low"].iloc[-5:].min()))
                - 0.15 * atr15
            )
            risk = entry - raw_sl
            if risk <= 0.0 or risk > self.CTR_MAX_STOP_ATR15 * atr15:
                return None
            risk = max(risk, 0.35 * atr15)
            sl = entry - risk
            reward = min(
                risk,
                entry * self.CTR_MAX_TP_PCT,
                float(tpc_gap["available_reward"]),
            )
            ema_reward = float(ema20.iloc[-1]) - entry
            if 0.80 * risk <= ema_reward <= reward:
                reward = ema_reward
            rr = reward / max(risk, 1e-12)
            if reward <= 0.0 or rr < self.CTR_MIN_RR:
                return None
            tp = entry + reward
            if tp >= float(tpc_gap["target_guard"]):
                return None
            trigger = f"CTR_{level_name}_HMA16_REJECTION_LONG"
            compat_trend = S.Trend.BEAR

        setup = self.strat._setup_from_context(context)
        if setup is None:
            return None
        structure_level = level
        room_pct = abs(tp - entry) / max(entry, 1e-12)
        reason = (
            f"CTR Counter-Trend Reversion {counter_side.value} | "
            f"primary 1H {primary.side.value} Trend {primary.score:.0f} "
            f"Q {quality.q:.0f} | 15M {level_name} rejection | "
            f"HMA16 flip | exhaustion {sum(bool(x) for x in exhaustion)}/4 | "
            f"TPC gap to {tpc_gap['name']} "
            f"{tpc_gap['gap_pct'] * 100:.2f}%/"
            f"{tpc_gap['gap_atr']:.2f}ATR | TP protected before TPC zone | "
            f"RR {rr:.2f} | margin {self.CTR_MARGIN_MULTIPLIER:.0%} of TPC"
        )
        return S.EntrySignal(
            side=counter_side,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            trend_4h=compat_trend,
            q_1h=quality.q,
            adx_1h=quality.adx,
            chop_1h=quality.chop,
            setup=setup,
            trigger=trigger,
            room_pct=room_pct,
            atr15=atr15,
            structure_level=structure_level,
            reason=reason,
        )

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
            half = max(
                self.strat.sr_touch_zone_atr5 * atr5,
                abs(level) * 1e-6,
            )
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
            gap = self._ctr_gap_status(df4h, df1h, df15)
            ctr = self._ctr_generate_entry(df4h, df1h, df15, df5)
            ctr_part = "CTR=READY" if ctr is not None else "CTR=WAIT"
            zone_part = f" | {zone}" if zone else ""
            self._view[symbol] = (
                f"5M px={px:.6g} | {self._asset_profile_status()}"
                f"{zone_part} | {q} | {status} | {gap} | {ctr_part}"
            )
        except Exception as exc:
            self._view[symbol] = f"view error: {str(exc)[:140]}"

    async def _look_for_entry(self, symbol: str, st: dict):
        self._apply_asset_profile(symbol)
        original_margin = float(self.cfg.margin_per_position_usd)
        try:
            # Preflight only determines sizing. The inherited production entry
            # path still performs all normal schedule, cooldown, order, chart,
            # reconciliation and position-limit checks.
            try:
                df5, df15, df1h, df4h = await self._entry_frames(symbol)
                preview = self.strat.generate_entry(
                    df4h, df1h, df15, df5, has_open_position=False
                )
                if preview is not None and str(preview.trigger).startswith("CTR_"):
                    self.cfg.margin_per_position_usd = max(
                        5.0,
                        original_margin * self.CTR_MARGIN_MULTIPLIER,
                    )
            except Exception as exc:
                _LOG.debug("[%s] CTR sizing preflight unavailable: %s", symbol, exc)
            return await super()._look_for_entry(symbol, st)
        finally:
            self.cfg.margin_per_position_usd = original_margin

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
                "Primary TPC: unchanged `1H Direction + Q → 15M S/R → 5M Hold/Reclaim`\n"
                "Quality: `ADX 45 + CHOP 35 + directional DMI 20`\n"
                "`Q ≥60` normal S1/S2 or R1/R2 hold/reclaim\n"
                "`Q 45–59` only S2/R2, or reclaim at S1/R1\n"
                "`Q <45` no trade\n"
                "CTR fallback: opposing S/R + 15M HMA16 flip + rejection + exhaustion `2/4`\n"
                "CTR gap: distance to pending TPC zone must be `≥0.9%` or `≥1.2 ATR15`\n"
                "CTR TP is capped at `0.7%` and must finish before the TPC zone\n"
                "CTR uses `40%` margin, stop ≤`1.0 ATR15`, RR ≥`0.8`\n"
                "CTR never opens when TPC has a valid signal or the symbol already has a position\n"
                "Asset profiles: execution-zone width + ATR stop only\n"
                "Prepared: `BTC ETH SOL HYPE XRP TRX XAU XAG CL`\n"
                "TPC Stage 1 `+0.7%→lock +0.4%` | Stage 2 `+1.1%→lock +0.75%` | Final TP `+1.5%`\n"
                "4H and Confidence are diagnostic only."
            )

        _LOG.info(
            "TPC Sentinel v1.0 startup complete: TPC primary unchanged; "
            "CTR TPC-zone-gap fallback active; multi-asset risk profiles active"
        )


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            signal_module = v15.v14.v13.v12.v11.v10.v9.v8.v7.v5.v4.v3.base._signal
            loop.add_signal_handler(
                getattr(signal_module, sig_name), bot.request_shutdown
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
