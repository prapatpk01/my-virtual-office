"""TPC Dynamic Zone V6.4 production runtime.

Keeps the current OKX execution, restart reconciliation, native SL/TP and
dynamic-zone infrastructure.  Primary TPC remains authoritative.  A small
counter-trend fallback is evaluated only when TPC has no valid signal.
"""
from __future__ import annotations

import asyncio
import math
import os
import time

import numpy as np

import main_v15 as v15
import strategy_v12 as S

_LOG = v15._LOG


class FastPrecisionStrategy(S.PrecisionTrendStructureV12):
    """Frequency-preserving EMA13 continuation on validated liquid symbols."""

    FAST_SYMBOLS = {
        item.strip().upper()
        for item in os.environ.get(
            "TPC_FAST_SYMBOLS",
            "BTC,ETH,SOL,HYPE,XRP,TRX,XAU,XAG,CL,DOGE",
        ).split(",")
        if item.strip()
    }
    FAST_TREND_MIN = float(os.environ.get("TPC_FAST_TREND_MIN", "68"))
    FAST_EDGE_MIN = float(os.environ.get("TPC_FAST_EDGE_MIN", "15"))
    FAST_Q_MIN = float(os.environ.get("TPC_FAST_Q_MIN", "58"))
    FAST_ADX_MIN = float(os.environ.get("TPC_FAST_ADX_MIN", "15"))
    FAST_CHOP_MAX = float(os.environ.get("TPC_FAST_CHOP_MAX", "62"))
    FAST_MAX_CHASE_ATR = float(os.environ.get("TPC_FAST_MAX_CHASE_ATR", "0.85"))

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.current_symbol = ""
        if "TPC_MIN_ROOM_ATR" not in os.environ:
            self.min_room_atr = 0.40
        if "TPC_MIN_RR" not in os.environ:
            self.min_rr = 1.30
        if "TPC_SL_MAX_PCT" not in os.environ:
            self.sl_max_pct = 0.012
        if "TPC_ZONE_REACTION_BARS" not in os.environ:
            self.zone_reaction_bars = max(self.zone_reaction_bars, 5)
        if "TPC_MAX_CHASE_ATR" not in os.environ:
            self.max_chase_atr = 1.25

    def evaluate(self, df4h, df1h, df15, df5):
        decision = super().evaluate(df4h, df1h, df15, df5)
        symbol = str(self.current_symbol or "").upper()
        if (
            decision.ready
            or decision.stage != "15M_TRIGGER"
            or symbol not in self.FAST_SYMBOLS
            or decision.direction.side is None
            or decision.quality is None
            or not isinstance(decision.context, S.DynamicContext)
            or decision.context.mode != "EMA13_FALLBACK"
            or decision.direction.score < self.FAST_TREND_MIN
            or decision.direction.edge < self.FAST_EDGE_MIN
            or decision.quality.q < self.FAST_Q_MIN
            or decision.quality.adx < self.FAST_ADX_MIN
            or decision.quality.chop > self.FAST_CHOP_MAX
        ):
            return decision

        side = decision.direction.side
        macro_ok, _, _ = self._macro_aligned(df4h, side)
        if not macro_ok:
            return decision

        d15 = self._prepared(df15)
        row = d15.iloc[-1]
        atr = float(row["atr"])
        if not np.isfinite(atr) or atr <= 0:
            return decision
        close = float(row["close"])
        open_ = float(row["open"])
        aligned = (
            close > float(row["ema13"])
            if side == S.Side.LONG
            else close < float(row["ema13"])
        )
        momentum = close > open_ if side == S.Side.LONG else close < open_
        body_atr = abs(close - open_) / atr
        chase = abs(close - float(row["ema13"])) / atr
        if (
            not aligned
            or not momentum
            or body_atr < self.min_body_atr
            or chase > self.FAST_MAX_CHASE_ATR
        ):
            return decision

        trigger = "15M_FAST_EMA13_CONTINUATION"
        provisional = S.DecisionState(
            True,
            "READY",
            "FAST_EMA13_CONTINUATION",
            decision.direction,
            decision.quality,
            decision.context,
            decision.setup_type,
            (trigger, atr),
            decision.direction.score,
        )
        risk = self._risk_plan(provisional, df15, df5)
        if risk is None or risk[-1] < self.min_rr:
            return decision
        return S.DecisionState(
            True,
            "READY",
            f"FAST_EMA13_CONTINUATION RR {risk[-1]:.2f}",
            decision.direction,
            decision.quality,
            decision.context,
            decision.setup_type,
            (trigger, atr),
            decision.direction.score,
        )


class Bot(v15.Bot):
    XAG_MIN_Q = 60.0
    XAG_ENTRY_START_UTC = 0
    XAG_ENTRY_END_UTC = 12

    # CTR master switch — DISABLED by default.
    # Backtest (BTC, Mar-May 2026, fee 0.05%) rejected every CTR variant:
    #   production flip -1.5R | momentum-decel -16.2R | 5M flip -17.1R |
    #   no-flip -61.3R  — a flat ~-1.1R per trade, so the loss scales linearly
    #   with trade count. Root cause: the CTR stop is only ~0.268% wide, so the
    #   0.10% round-trip fee eats 0.37R of every trade; at RR 1.86 it needs a
    #   ~48% win rate but counter-trend entries only won ~20%.
    # The engine is intentionally KEPT (not deleted) pending a redesign with a
    # wider stop/target so fee drag stops dominating. Set CTR_ENABLED=1 to
    # re-enable once a variant actually backtests positive.
    CTR_ENABLED = os.environ.get("CTR_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )

    CTR_MARGIN_MULTIPLIER = 0.40
    CTR_MIN_TREND = 55.0
    CTR_MIN_Q = 50.0
    CTR_RECENT_ZONE_BARS = 6
    CTR_MIN_GAP_PCT = 0.007
    CTR_MIN_GAP_ATR = 1.00
    CTR_MAX_TP_PCT = 0.007
    CTR_MAX_STOP_ATR = 1.20
    CTR_MIN_RR = 0.70
    CTR_ZONE_BUFFER_ATR = 0.10

    def __init__(self):
        super().__init__()
        self.strat = FastPrecisionStrategy(self.cfg.strategy_config())

        # Nothing is disabled by default. Railway may still explicitly set a
        # comma-separated TPC_DISABLED_SYMBOLS value when required.
        disabled = os.environ.get("TPC_DISABLED_SYMBOLS", "")
        self.disabled_entry_symbols = {
            item.strip().upper() for item in disabled.split(",") if item.strip()
        }
        self._closed_seen = {
            symbol: not bool((self.state.get(symbol) or {}).get("pos"))
            for symbol in self.cfg.symbols
        }
        cooldown_min = float(os.environ.get("TPC_POST_CLOSE_COOLDOWN_MIN", "20"))
        self.post_close_cooldown_sec = max(0.0, cooldown_min * 60.0)
        self.risk_per_trade_pct = float(
            os.environ.get("FAST_RISK_PER_TRADE_PCT", "0.02")
        )
        self.min_dynamic_margin = float(
            os.environ.get("FAST_MIN_MARGIN_USD", "5.0")
        )
        self._shutdown_requested = False
        self._client_closed = False
        self._entry_symbol = ""
        self._xag_filter_reason = "NOT_EVALUATED"
        self._ctr_reason: dict[str, str] = {}

        self._raw_generate_entry = self.strat.generate_entry

        def combined_generate_entry(
            df4h, df1h, df15, df5, has_open_position: bool = False
        ):
            self.strat.current_symbol = self._base_symbol(self._entry_symbol)
            signal = self._raw_generate_entry(
                df4h,
                df1h,
                df15,
                df5,
                has_open_position=has_open_position,
            )
            if signal is None and not has_open_position and self.CTR_ENABLED:
                signal = self._ctr_generate_entry(df4h, df1h, df15, df5)
            return self._apply_symbol_entry_filter(
                self._entry_symbol,
                signal,
                df15,
            )

        self.strat.generate_entry = combined_generate_entry

    @staticmethod
    def _base_symbol(symbol: str) -> str:
        return str(symbol or "").upper().split("/", 1)[0].split(":", 1)[0]

    @staticmethod
    def _closed_candle_utc_hour(frame) -> int:
        if frame is None or len(frame) == 0:
            return -1
        timestamp = frame.index[-1]
        try:
            if getattr(timestamp, "tzinfo", None) is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            return int(timestamp.hour)
        except (AttributeError, TypeError, ValueError):
            return -1

    def _apply_symbol_entry_filter(self, symbol: str, signal, df15):
        """Apply the validated XAG session/Q filter to both TPC and CTR."""
        if self._base_symbol(symbol) != "XAG":
            return signal
        if signal is None:
            self._xag_filter_reason = "WAIT_SIGNAL"
            return None
        if float(signal.q_1h) < self.XAG_MIN_Q:
            self._xag_filter_reason = (
                f"Q_{float(signal.q_1h):.1f}_LT_{self.XAG_MIN_Q:.0f}"
            )
            return None

        is_ctr = str(signal.trigger or "").startswith("CTR_")
        if (
            not is_ctr
            and "EMA13_TREND_PULLBACK" not in str(signal.reason or "").upper()
        ):
            self._xag_filter_reason = "NEED_EMA13_PULLBACK_LOCATION"
            return None

        hour = self._closed_candle_utc_hour(df15)
        session_open = self.XAG_ENTRY_START_UTC <= hour < self.XAG_ENTRY_END_UTC
        if not session_open:
            self._xag_filter_reason = f"SESSION_CLOSED_{hour:02d}UTC"
            return None

        engine = "CTR" if is_ctr else "TPC"
        self._xag_filter_reason = (
            f"PASS_{engine}_Q{float(signal.q_1h):.0f}_{hour:02d}UTC"
        )
        return signal

    @staticmethod
    def _rsi14(close) -> float:
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        value = float(rsi.iloc[-1])
        return value if math.isfinite(value) else 50.0

    @staticmethod
    def _rejection(row, side: S.Side) -> bool:
        open_ = float(row["open"])
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        body = abs(close - open_)
        candle_range = max(high - low, 1e-12)
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        if side == S.Side.SHORT:
            return close < open_ or upper_wick >= max(body, 0.30 * candle_range)
        return close > open_ or lower_wick >= max(body, 0.30 * candle_range)

    def _ctr_block(self, reason: str):
        symbol = self._base_symbol(self._entry_symbol)
        self._ctr_reason[symbol] = reason
        return None

    def _ctr_generate_entry(self, df4h, df1h, df15, df5):
        """Small counter-trend trade while price travels toward the TPC zone."""
        if len(df1h) < 60 or len(df15) < 70 or len(df5) < 40:
            return self._ctr_block("WARMUP")

        primary, quality = self.strat._direction_1h(df1h)
        if primary.side is None:
            return self._ctr_block("NO_1H_DIRECTION")
        if primary.score < self.CTR_MIN_TREND or quality.q < self.CTR_MIN_Q:
            return self._ctr_block(
                f"TREND_Q_LOW({primary.score:.0f}/{quality.q:.0f})"
            )

        counter_side = (
            S.Side.SHORT if primary.side == S.Side.LONG else S.Side.LONG
        )
        d15 = self.strat._prepared(df15)
        atr = float(d15["atr"].iloc[-1])
        if not math.isfinite(atr) or atr <= 0:
            return self._ctr_block("ATR_INVALID")

        current = d15.iloc[-1]
        previous = d15.iloc[-2]
        entry = float(current["close"])
        counter_context = self.strat._zone_context(d15, counter_side)
        counter_zone = counter_context.active_zone
        if counter_zone is None:
            return self._ctr_block("NO_COUNTER_ZONE")

        recent = d15.iloc[-self.CTR_RECENT_ZONE_BARS :]
        touch_buffer = self.strat.zone_touch_atr * atr
        if counter_side == S.Side.SHORT:
            touched = float(recent["high"].max()) >= counter_zone.lower - touch_buffer
            left_zone = entry <= counter_zone.upper + touch_buffer
        else:
            touched = float(recent["low"].min()) <= counter_zone.upper + touch_buffer
            left_zone = entry >= counter_zone.lower - touch_buffer
        if not touched or not left_zone:
            return self._ctr_block("WAIT_RECENT_ZONE_TOUCH")

        hma_now = float(current["hma16"])
        hma_prev = float(previous["hma16"])
        if counter_side == S.Side.SHORT:
            crossed = float(previous["close"]) >= hma_prev and entry < hma_now
            turned = hma_now < hma_prev
            hma_ok = entry < hma_now and (crossed or turned)
        else:
            crossed = float(previous["close"]) <= hma_prev and entry > hma_now
            turned = hma_now > hma_prev
            hma_ok = entry > hma_now and (crossed or turned)
        if not hma_ok:
            return self._ctr_block("WAIT_HMA16_CROSS_OR_TURN")

        rejection = self._rejection(current, counter_side) or self._rejection(
            previous, counter_side
        )
        if not rejection:
            return self._ctr_block("WAIT_REJECTION")

        primary_context = self.strat._zone_context(d15, primary.side)
        target_zone = primary_context.active_zone or counter_context.opposing_zone
        ema13 = float(current["ema13"])
        if primary.side == S.Side.LONG:
            if target_zone is not None:
                target_edge = float(target_zone.upper)
                target_name = target_zone.label
            else:
                target_edge = ema13
                target_name = "EMA13"
            if target_edge >= entry:
                return self._ctr_block("TPC_TARGET_NOT_BELOW")
            gap = entry - target_edge
            guard = target_edge + self.CTR_ZONE_BUFFER_ATR * atr
            available_reward = entry - guard
        else:
            if target_zone is not None:
                target_edge = float(target_zone.lower)
                target_name = target_zone.label
            else:
                target_edge = ema13
                target_name = "EMA13"
            if target_edge <= entry:
                return self._ctr_block("TPC_TARGET_NOT_ABOVE")
            gap = target_edge - entry
            guard = target_edge - self.CTR_ZONE_BUFFER_ATR * atr
            available_reward = guard - entry

        gap_pct = gap / max(entry, 1e-12)
        gap_atr = gap / max(atr, 1e-12)
        if gap_pct < self.CTR_MIN_GAP_PCT and gap_atr < self.CTR_MIN_GAP_ATR:
            return self._ctr_block(
                f"TPC_GAP_SMALL({gap_pct * 100:.2f}%/{gap_atr:.2f}ATR)"
            )
        if available_reward <= 0:
            return self._ctr_block("NO_TARGET_ROOM")

        close = d15["close"].astype(float)
        basis = close.rolling(20).mean()
        std = close.rolling(20).std(ddof=0)
        upper = float((basis + 2.0 * std).iloc[-1])
        lower = float((basis - 2.0 * std).iloc[-1])
        rsi = self._rsi14(close)
        recent_high = float(recent["high"].max())
        recent_low = float(recent["low"].min())

        if counter_side == S.Side.SHORT:
            exhaustion = [
                (recent_high - ema13) / atr >= 1.00,
                (recent_high - hma_now) / atr >= 0.60,
                rsi >= 65.0,
                recent_high >= upper,
            ]
            raw_sl = max(counter_zone.upper, recent_high) + 0.15 * atr
            risk = raw_sl - entry
        else:
            exhaustion = [
                (ema13 - recent_low) / atr >= 1.00,
                (hma_now - recent_low) / atr >= 0.60,
                rsi <= 35.0,
                recent_low <= lower,
            ]
            raw_sl = min(counter_zone.lower, recent_low) - 0.15 * atr
            risk = entry - raw_sl

        required_exhaustion = 2 if quality.q >= 90.0 else 1
        exhaustion_count = sum(bool(value) for value in exhaustion)
        if exhaustion_count < required_exhaustion:
            return self._ctr_block(
                f"EXHAUSTION_{exhaustion_count}_LT_{required_exhaustion}"
            )
        if risk <= 0 or risk > self.CTR_MAX_STOP_ATR * atr:
            return self._ctr_block(f"STOP_TOO_WIDE({risk / atr:.2f}ATR)")

        risk = max(risk, 0.30 * atr)
        reward = min(
            risk,
            entry * self.CTR_MAX_TP_PCT,
            available_reward,
        )
        rr = reward / max(risk, 1e-12)
        if reward <= 0 or rr < self.CTR_MIN_RR:
            return self._ctr_block(f"RR_LOW({rr:.2f})")

        if counter_side == S.Side.SHORT:
            sl = entry + risk
            tp = entry - reward
            if tp <= guard:
                return self._ctr_block("TP_CROSSES_TPC_ZONE")
            trend = S.Trend.BEAR
        else:
            sl = entry - risk
            tp = entry + reward
            if tp >= guard:
                return self._ctr_block("TP_CROSSES_TPC_ZONE")
            trend = S.Trend.BULL

        trigger = (
            f"CTR_{counter_zone.kind.upper()}_RECENT_TOUCH_HMA16_"
            f"{counter_side.value}"
        )
        reason = (
            f"CTR {counter_side.value} | primary 1H {primary.side.value} "
            f"Trend {primary.score:.0f} Q {quality.q:.0f} | "
            f"recent {counter_zone.label} touch → HMA16 cross/turn | "
            f"exhaustion {exhaustion_count}/4 | TPC gap to {target_name} "
            f"{gap_pct * 100:.2f}%/{gap_atr:.2f}ATR | "
            f"RR {rr:.2f} | margin {self.CTR_MARGIN_MULTIPLIER:.0%}"
        )
        self._ctr_reason[self._base_symbol(self._entry_symbol)] = "READY"
        return S.EntrySignal(
            side=counter_side,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            trend_4h=trend,
            q_1h=quality.q,
            adx_1h=quality.adx,
            chop_1h=quality.chop,
            setup=S.SetupType.PULLBACK,
            trigger=trigger,
            room_pct=reward / max(entry, 1e-12),
            atr15=atr,
            structure_level=(
                counter_zone.upper
                if counter_side == S.Side.SHORT
                else counter_zone.lower
            ),
            reason=reason,
        )

    def request_shutdown(self) -> None:
        """Stop scheduling work; keep the OKX client alive for in-flight work."""
        self._shutdown_requested = True
        self._running = False
        _LOG.info("TPC-ZONE-V6.4 graceful shutdown requested")

    async def run_forever(self):
        """Finish the current symbol safely before closing exchange access."""
        while self._running and not self._shutdown_requested:
            for symbol in self.cfg.symbols:
                if self._shutdown_requested:
                    break
                try:
                    await self._process(symbol)
                except Exception as exc:
                    if (
                        self._shutdown_requested
                        and "closed by the user" in str(exc).lower()
                    ):
                        break
                    _LOG.error("[%s] unhandled: %s", symbol, exc, exc_info=True)
            if self._shutdown_requested:
                break
            self._maybe_status_log()
            await asyncio.sleep(self.cfg.poll_interval_sec)

    async def stop(self):
        """Close the shared OKX client exactly once."""
        self._running = False
        self._shutdown_requested = True
        if self._client_closed:
            return
        self._client_closed = True
        await self.client.close()
        _LOG.info("TPC-ZONE-V6.4 shutdown complete")

    def _set_view_v3(self, symbol: str, df5, df15, df1h, df4h):
        try:
            self._entry_symbol = symbol
            if self.open_position_count() >= self.cfg.max_positions:
                self._view[symbol] = (
                    f"TPC-ZONE-V6.4 POSITION LIMIT | MAX {self.cfg.max_positions}"
                )
                return
            remaining = max(0, self._cooldown_until.get(symbol, 0) - time.time())
            if remaining > 0:
                self._view[symbol] = (
                    f"TPC-ZONE-V6.4 COOLDOWN | {remaining / 60:.0f}m"
                )
                return

            px = float(df15["close"].iloc[-1]) if len(df15) else 0.0
            self.strat.current_symbol = self._base_symbol(symbol)
            status = self.strat.entry_status(df4h, df1h, df15, df5)
            tpc_preview = self._raw_generate_entry(
                df4h,
                df1h,
                df15,
                df5,
                has_open_position=False,
            )
            if not self.CTR_ENABLED:
                ctr_status = "CTR=OFF"
            elif tpc_preview is not None:
                ctr_status = "CTR=SKIP_TPC_READY"
            else:
                ctr_preview = self._ctr_generate_entry(df4h, df1h, df15, df5)
                reason = self._ctr_reason.get(
                    self._base_symbol(symbol),
                    "WAIT",
                )
                ctr_status = (
                    "CTR=READY" if ctr_preview is not None else f"CTR=WAIT:{reason}"
                )

            xag_status = ""
            if self._base_symbol(symbol) == "XAG":
                self.strat.generate_entry(
                    df4h,
                    df1h,
                    df15,
                    df5,
                    has_open_position=False,
                )
                xag_status = f" | XAGFilter={self._xag_filter_reason}"
            self._view[symbol] = (
                f"15M px={px:.6g} | {status} | {ctr_status}{xag_status}"
            )
        except Exception as exc:
            self._view[symbol] = f"TPC-ZONE-V6.4 view error: {str(exc)[:140]}"

    async def _manage(self, symbol: str, st: dict):
        had_position = bool(st.get("pos"))
        pos = st.get("pos") or {}
        if pos.get("recovery_quarantine"):
            side = str(pos.get("side") or "")
            native_sl, native_tp = await self.client.fetch_attached_stops(
                symbol,
                side,
            )
            if native_sl and native_tp:
                pos.update(
                    {
                        "sl": float(native_sl),
                        "initial_sl": float(native_sl),
                        "tp": float(native_tp),
                        "risk": abs(
                            float(pos.get("entry") or 0) - float(native_sl)
                        ),
                        "recovery_quarantine": False,
                    }
                )
                self._save_state()
                _LOG.info("[%s] recovery quarantine cleared read-only", symbol)
        await super()._manage(symbol, st)
        has_position = bool(st.get("pos"))
        if had_position and not has_position:
            self._cooldown_until[symbol] = (
                time.time() + self.post_close_cooldown_sec
            )
            self._closed_seen[symbol] = True
            _LOG.info(
                "[%s] TPC-ZONE-V6.4 post-close cooldown %.0f minutes",
                symbol,
                self.post_close_cooldown_sec / 60.0,
            )

    async def _reconcile_startup(self):
        """Recover positions without cancelling or replacing existing TP/SL."""
        await super()._reconcile_startup()
        for symbol in self.cfg.symbols:
            st = self.state.get(symbol) or {}
            pos = st.get("pos") or {}
            side = str(pos.get("side") or "")
            entry = float(pos.get("entry") or 0.0)
            amount = float(pos.get("amount") or 0.0)
            if side not in ("long", "short") or entry <= 0 or amount <= 0:
                continue

            native_sl = native_tp = None
            for _ in range(6):
                native_sl, native_tp = await self.client.fetch_attached_stops(
                    symbol,
                    side,
                )
                if native_sl and native_tp:
                    break
                await asyncio.sleep(1.0)

            if native_sl and native_tp:
                sl = float(native_sl)
                tp = float(native_tp)
                pos.update(
                    {
                        "sl": sl,
                        "initial_sl": sl,
                        "tp": tp,
                        "risk": abs(entry - sl),
                        "recovery_quarantine": False,
                    }
                )
                _LOG.info(
                    "[%s] recovered position kept existing protection: "
                    "SL %.8g TP %.8g",
                    symbol,
                    sl,
                    tp,
                )
            else:
                pos["recovery_quarantine"] = True
                _LOG.warning(
                    "[%s] recovered protection read unavailable",
                    symbol,
                )
                await self.tg.send_text(
                    f"⚠️ `{symbol}` recovered TP/SL could not be read yet. "
                    "No OKX order was cancelled, replaced or added."
                )
        self._save_state()

    async def _look_for_entry(self, symbol: str, st: dict):
        """Risk-size TPC/CTR, then verify attached protection read-only."""
        self._entry_symbol = symbol
        had_position = bool(st.get("pos"))
        base_symbol = self._base_symbol(symbol)
        if not had_position and base_symbol in self.disabled_entry_symbols:
            self._view[symbol] = (
                f"TPC-ZONE-V6.4 ENTRY DISABLED | {base_symbol}"
            )
            return

        configured_margin = float(self.cfg.margin_per_position_usd)
        try:
            try:
                df5, df15, df1h, df4h = await self._entry_frames(symbol)
                preview = self.strat.generate_entry(
                    df4h,
                    df1h,
                    df15,
                    df5,
                    has_open_position=False,
                )
                if preview is not None and preview.entry_price > 0:
                    stop_pct = abs(
                        float(preview.entry_price) - float(preview.stop_loss)
                    ) / float(preview.entry_price)
                    balance = await self.client.fetch_balance_usdt()
                    is_ctr = str(preview.trigger or "").startswith("CTR_")
                    risk_multiplier = (
                        self.CTR_MARGIN_MULTIPLIER if is_ctr else 1.0
                    )
                    risk_budget = max(
                        0.0,
                        balance * self.risk_per_trade_pct * risk_multiplier,
                    )
                    margin = risk_budget / max(
                        stop_pct * float(self.cfg.leverage),
                        1e-12,
                    )
                    margin_cap = configured_margin * risk_multiplier
                    self.cfg.margin_per_position_usd = min(
                        margin_cap,
                        max(self.min_dynamic_margin, margin),
                    )
                    _LOG.info(
                        "[%s] %s dynamic risk size: balance=%.2f risk=$%.2f "
                        "SL=%.2f%% margin=$%.2f cap=$%.2f",
                        symbol,
                        "CTR" if is_ctr else "TPC",
                        balance,
                        risk_budget,
                        stop_pct * 100,
                        self.cfg.margin_per_position_usd,
                        margin_cap,
                    )
            except Exception as exc:
                _LOG.warning(
                    "[%s] dynamic sizing preflight failed: %s",
                    symbol,
                    exc,
                )
            await super()._look_for_entry(symbol, st)
        finally:
            self.cfg.margin_per_position_usd = configured_margin

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

        native_sl = native_tp = None
        for _ in range(8):
            native_sl, native_tp = await self.client.fetch_attached_stops(
                symbol,
                side,
            )
            if native_sl and native_tp:
                break
            await asyncio.sleep(1.0)
        if native_sl and native_tp:
            _LOG.info(
                "[%s] attached protection visible: SL %.8g TP %.8g",
                symbol,
                native_sl,
                native_tp,
            )
            return

        _LOG.warning(
            "[%s] attached protection not visible yet; no action taken",
            symbol,
        )
        await self.tg.send_text(
            f"⚠️ `{symbol}` TP/SL is not visible through the read API yet. "
            "The position remains open and no OKX order was changed."
        )

    async def start(self):
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start: " + "; ".join(problems))
        if not self.cfg.paper and not await self.client.ensure_hedge_mode():
            raise RuntimeError("Could not confirm OKX hedge mode.")

        balance = await self.client.fetch_balance_usdt()
        _LOG.info(
            "=== TPC DYNAMIC ZONE V6.4 [%s] symbols=%s margin=$%.2f "
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
                f"🎯 *TPC Dynamic Zone V6.4 — {mode}*\n"
                f"Symbols: `{', '.join(self.cfg.symbols)}`\n"
                f"Balance: `{balance:.2f}` USDT | Margin cap "
                f"`${self.cfg.margin_per_position_usd:.2f}` | "
                f"Leverage `x{self.cfg.leverage}` | "
                f"Max `{self.cfg.max_positions}`\n\n"
                "Primary TPC: `1H direction/Q → 15M dynamic zone or EMA13 pullback → closed-15M trigger`\n"
                "TPC trigger: `HMA16 flip, EMA13 reclaim, or strong fast EMA13 continuation`\n"
                "Fast symbols: `BTC ETH SOL HYPE XRP TRX XAU XAG CL DOGE`\n"
                "TPC defaults: `room≥0.40ATR`, `RR≥1.30`, `SL≤1.20%`\n"
                + (
                    "CTR fallback: only when TPC has no signal; recent supply/demand touch + HMA16 cross/turn + rejection\n"
                    "CTR gap: `≥0.7% or ≥1.0ATR` toward the pending TPC zone\n"
                    "CTR risk: `40% sizing`, `TP≤0.7%`, `SL≤1.2ATR`, `RR≥0.7`\n"
                    if self.CTR_ENABLED else
                    "CTR fallback: `DISABLED` (backtest-negative; set `CTR_ENABLED=1` to re-enable)\n"
                ) +
                "Entries disabled by default: `none`\n"
                f"Explicit disabled symbols: `{', '.join(sorted(self.disabled_entry_symbols)) or 'none'}`\n"
                f"Re-entry cooldown: `{self.post_close_cooldown_sec / 60:.0f} minutes`\n"
                "Recovery: existing positions and native SL/TP reconciled after restart"
            )
        _LOG.info("TPC Dynamic Zone V6.4 startup complete")


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            signal_module = (
                v15.v14.v13.v12.v11.v10.v9.v8.v7.v5.v4.v3.base._signal
            )
            loop.add_signal_handler(
                getattr(signal_module, sig_name),
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
