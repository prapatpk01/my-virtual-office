"""Guaranteed entry-alert delivery for EMA Hybrid A-E.

The inherited HMA entry path persists a newly opened position before rendering /
sending its Telegram chart.  That is correct for trade safety, but a downstream
chart or Telegram failure can leave a real/open PAPER position with no visible
entry notification.  This mixin makes notification delivery observable and
recovers it without changing entry, risk, TP or SL logic.
"""
from __future__ import annotations

import os
import re
import time

from chart_engine import build_entry_chart


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _setup_key(trigger: str) -> str:
    t = str(trigger or "").upper()
    if "MACD_VOLUME_" in t:
        return "E_MACD_VOLUME"
    if "MA5_MA20_" in t:
        return "D_MA5_MA20"
    if "BOLL_MACD_KDJ_" in t:
        return "C_BOLL_MACD_KDJ"
    if "EMA5M_CROSS_" in t:
        return "A_EMA_CROSS"
    if "PULLBACK_" in t:
        return "B_PULLBACK_RECLAIM"
    return "UNKNOWN"


def _setup_label(trigger: str) -> str:
    return {
        "A_EMA_CROSS": "A · EMA8/13 CROSS",
        "B_PULLBACK_RECLAIM": "B · PULLBACK RECLAIM/BOS",
        "C_BOLL_MACD_KDJ": "C · BOLL+MACD+KDJ REVERSAL",
        "D_MA5_MA20": "D · MA5/MA20 TREND PULLBACK",
        "E_MACD_VOLUME": "E · MACD+VOLUME CONFIRM",
    }.get(_setup_key(trigger), "UNKNOWN")


class EntryAlertGuardMixin:
    """Guarantee that every newly opened A-E position gets an entry alert."""

    ENTRY_ALERT_DELIVERY_WINDOW_SEC = float(
        os.getenv("EMA_ENTRY_ALERT_DELIVERY_WINDOW_SEC", "120")
    )
    ENTRY_ALERT_RECOVER_EXISTING_E = _env_bool(
        "EMA_ENTRY_ALERT_RECOVER_EXISTING_E", True
    )
    ENTRY_ALERT_EXISTING_MAX_AGE_SEC = float(
        os.getenv("EMA_ENTRY_ALERT_EXISTING_MAX_AGE_SEC", str(6 * 60 * 60))
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ema_entry_delivery: dict[tuple[str, str], float] = {}
        self._install_entry_delivery_guard()

    @staticmethod
    def _root(symbol: str) -> str:
        return str(symbol or "").split("/")[0].upper()

    def _entry_identity_from_text(self, text: str):
        if not isinstance(text, str):
            return None
        plain = text.replace("`", "").replace("*", "")
        upper = plain.upper()
        # Only treat true entry captions as delivery confirmations.  TP/SL/status
        # messages may also contain a symbol/side but do not contain this pair.
        if "STRUCTURE SL" not in upper or "MARGIN $" not in upper:
            return None
        for symbol in getattr(self.cfg, "symbols", []):
            root = self._root(symbol)
            for side in ("LONG", "SHORT"):
                if re.search(rf"\b{re.escape(root)}\s+{side}\b", upper):
                    return root, side.lower()
        return None

    def _mark_entry_delivery(self, text: str) -> None:
        ident = self._entry_identity_from_text(text)
        if ident is not None:
            self._ema_entry_delivery[ident] = time.monotonic()

    def _delivery_recent(self, symbol: str, side: str) -> bool:
        ident = (self._root(symbol), str(side or "").lower())
        ts = self._ema_entry_delivery.get(ident)
        return ts is not None and time.monotonic() - ts <= self.ENTRY_ALERT_DELIVERY_WINDOW_SEC

    def _install_entry_delivery_guard(self) -> None:
        """Track success and force text fallback when photo delivery returns False."""
        previous_text = self.tg.send_text
        previous_photo = self.tg._send_photo

        async def guarded_text(text: str) -> bool:
            try:
                result = previous_text(text)
                if hasattr(result, "__await__"):
                    result = await result
                ok = bool(result)
            except Exception as exc:
                self._entry_alert_log(
                    "error", "Telegram text delivery exception: %s", exc,
                    exc_info=True,
                )
                return False
            if ok:
                self._mark_entry_delivery(text)
            return ok

        async def guarded_photo(path: str, caption: str) -> bool:
            ok = False
            try:
                result = previous_photo(path, caption)
                if hasattr(result, "__await__"):
                    result = await result
                ok = bool(result)
            except Exception as exc:
                self._entry_alert_log(
                    "error", "Telegram photo delivery exception: %s", exc,
                    exc_info=True,
                )

            if not ok:
                self._entry_alert_log(
                    "warning",
                    "entry photo delivery returned false; forcing text fallback",
                )
                try:
                    result = previous_text(caption)
                    if hasattr(result, "__await__"):
                        result = await result
                    ok = bool(result)
                except Exception as exc:
                    self._entry_alert_log(
                        "error", "entry text fallback exception: %s", exc,
                        exc_info=True,
                    )
                    ok = False

            if ok:
                self._mark_entry_delivery(caption)
            return ok

        self.tg.send_text = guarded_text
        self.tg._send_photo = guarded_photo

    def _entry_alert_log(self, level: str, message: str, *args, **kwargs) -> None:
        logger = getattr(self, "_ema_alert_logger", None)
        if logger is None:
            # Every inherited EMA runtime exposes its logger through the class
            # module; standard print is intentionally avoided in production.
            try:
                import logging
                logger = logging.getLogger("precision_structure")
            except Exception:
                return
        fn = getattr(logger, level, None)
        if callable(fn):
            fn("[EMA ENTRY ALERT] " + message, *args, **kwargs)

    def _correct_position_setup(self, pos: dict) -> bool:
        trigger = str(pos.get("trigger") or "")
        key = _setup_key(trigger)
        if key == "UNKNOWN":
            return False
        if str(pos.get("setup") or "") == key:
            return False
        pos["setup"] = key
        return True

    def _recovery_caption(self, symbol: str, pos: dict) -> str:
        side = str(pos.get("side") or "?").upper()
        entry = float(pos.get("entry") or 0.0)
        sl = float(pos.get("sl") or 0.0)
        tp = float(pos.get("tp") or 0.0)
        trigger = str(pos.get("trigger") or "—")
        setup = _setup_label(trigger)
        margin = float(pos.get("margin_usd") or getattr(self.cfg, "margin_per_position_usd", 0.0) or 0.0)
        leverage = int(pos.get("leverage") or getattr(self.cfg, "leverage", 0) or 0)
        tp1 = "DONE" if pos.get("tp1_done") else "WAIT"
        return (
            f"⚠️ ENTRY ALERT RECOVERY\n"
            f"{'🟢' if side == 'LONG' else '🔴'} {self._root(symbol)} {side} @ {entry:.8g}\n"
            f"Setup Engine: {setup}\n"
            f"Trigger: {trigger}\n"
            f"EMA Hybrid A+B+C+D+E QUALITY V2.5\n"
            f"Structure SL {sl:.8g} | TP2 Liquidity/Swing {tp:.8g}\n"
            f"TP1 status: {tp1}\n"
            f"Margin ${margin:.2f} × x{leverage}\n"
            f"Reason: original entry chart/Telegram delivery was not confirmed; position management remained active."
        )

    async def _send_entry_recovery_alert(self, symbol: str, pos: dict) -> bool:
        caption = self._recovery_caption(symbol, pos)
        chart = None
        try:
            df5, _df15, _df1h, _df4h = await self._entry_frames(symbol)
            if len(df5):
                side = str(pos.get("side") or "").upper()
                entry = float(pos.get("entry") or 0.0)
                sl = float(pos.get("sl") or 0.0)
                tp = float(pos.get("tp") or 0.0)
                chart = build_entry_chart(
                    symbol,
                    df5,
                    side,
                    entry,
                    sl,
                    tp,
                    tp,
                    ema_fast_len=8,
                    ema_slow_len=13,
                    tf_label="5M",
                )
        except Exception as exc:
            self._entry_alert_log(
                "warning", "recovery chart build failed for %s: %s", symbol, exc,
                exc_info=True,
            )

        ok = False
        if chart:
            try:
                ok = bool(await self.tg._send_photo(chart, caption))
            except Exception as exc:
                self._entry_alert_log(
                    "error", "recovery photo send failed for %s: %s", symbol, exc,
                    exc_info=True,
                )

        if not ok:
            try:
                ok = bool(await self.tg.send_text(caption))
            except Exception as exc:
                self._entry_alert_log(
                    "error", "recovery text send failed for %s: %s", symbol, exc,
                    exc_info=True,
                )

        if ok:
            pos["ema_entry_alerted"] = True
            pos["ema_entry_alert_recovered"] = True
            self._save_state()
            self._entry_alert_log(
                "warning", "recovered missing entry alert for %s setup=%s trigger=%s",
                symbol, _setup_key(str(pos.get("trigger") or "")), pos.get("trigger"),
            )
        return ok

    async def _look_for_entry(self, symbol: str, st: dict):
        """Run inherited entry execution then verify visible delivery A-E."""
        before_open = bool(st.get("pos"))
        try:
            result = await super()._look_for_entry(symbol, st)
        except Exception as exc:
            pos = st.get("pos") or {}
            if not before_open and pos:
                self._entry_alert_log(
                    "error",
                    "post-order entry pipeline failed for %s; recovering alert: %s",
                    symbol,
                    exc,
                    exc_info=True,
                )
                if self._correct_position_setup(pos):
                    self._save_state()
                if not self._delivery_recent(symbol, pos.get("side")):
                    await self._send_entry_recovery_alert(symbol, pos)
                return None
            raise

        pos = st.get("pos") or {}
        if not before_open and pos:
            changed = self._correct_position_setup(pos)
            if self._delivery_recent(symbol, pos.get("side")):
                if not pos.get("ema_entry_alerted"):
                    pos["ema_entry_alerted"] = True
                    changed = True
            else:
                self._entry_alert_log(
                    "warning",
                    "position opened but no confirmed entry delivery: %s setup=%s trigger=%s",
                    symbol,
                    _setup_key(str(pos.get("trigger") or "")),
                    pos.get("trigger"),
                )
                await self._send_entry_recovery_alert(symbol, pos)
            if changed:
                self._save_state()
        return result

    async def _manage(self, symbol: str, st: dict):
        """One-time recovery for a recent pre-fix silent Setup-E position."""
        pos = st.get("pos") or {}
        trigger = str(pos.get("trigger") or "").upper()
        if (
            self.ENTRY_ALERT_RECOVER_EXISTING_E
            and pos
            and trigger.startswith("MACD_VOLUME_")
            and not pos.get("ema_entry_alerted")
            and not pos.get("ema_entry_recovery_attempted")
        ):
            pos["ema_entry_recovery_attempted"] = True
            self._save_state()
            opened_ms = int(pos.get("opened_ms") or 0)
            age_sec = max(0.0, time.time() - opened_ms / 1000.0) if opened_ms else 0.0
            if opened_ms and age_sec <= self.ENTRY_ALERT_EXISTING_MAX_AGE_SEC:
                await self._send_entry_recovery_alert(symbol, pos)
        return await super()._manage(symbol, st)
