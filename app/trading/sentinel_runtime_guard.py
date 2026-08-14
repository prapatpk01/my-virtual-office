"""Runtime safety for Sentinel-only production.

Two guarantees:
1) Every close notification is sent as plain text with retries so Telegram
   Markdown parsing cannot silently discard an EXIT alert.
2) After a Sentinel position closes, that symbol enters a configurable
   re-entry cooldown (default 120 minutes).  This prevents immediate whipsaw
   re-entry while still allowing the bot to manage other symbols normally.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

logger = logging.getLogger("sentinel_runtime_guard")


def _install_reliable_close_notifier() -> None:
    from .telegram_notifier import TelegramNotifier

    if getattr(TelegramNotifier, "_sentinel_close_alert_patch", False):
        return
    TelegramNotifier._sentinel_close_alert_patch = True

    async def _send_close_with_retry(self, text: str) -> None:
        for attempt in range(1, 4):
            try:
                ok = await self._send(text, parse_mode="")
                if ok:
                    logger.info("[SENTINEL-CLOSE] Telegram EXIT delivered attempt=%d", attempt)
                    return
            except Exception as exc:
                logger.warning("[SENTINEL-CLOSE] Telegram EXIT attempt %d failed: %s", attempt, exc)
            await asyncio.sleep(1.5 * attempt)
        logger.error("[SENTINEL-CLOSE] Telegram EXIT failed after 3 attempts")

    def notify_trade_closed(self, symbol: str, outcome: dict, stats: dict):
        if not getattr(self, "_enabled", False):
            return

        pnl = float(outcome.get("pnl_usd", 0.0) or 0.0)
        pnl_r = float(outcome.get("pnl_r", 0.0) or 0.0)
        pnl_pct = float(outcome.get("pnl_pct", 0.0) or 0.0)
        won = bool(outcome.get("won", pnl > 0))
        side = str(outcome.get("side") or "").upper()
        entry = float(outcome.get("entry", 0.0) or 0.0)
        exit_px = float(outcome.get("exit", 0.0) or 0.0)
        reason = str(outcome.get("reason") or outcome.get("reason_label") or "Position Closed")
        label = str(outcome.get("reason_label") or "Position Closed")
        sl = outcome.get("sl")
        tp = outcome.get("tp")
        fill = outcome.get("fill") or {}

        result = "✅" if won else "❌"
        sl_txt = f"{float(sl):,.4f}" if sl is not None else "—"
        tp_txt = f"{float(tp):,.4f}" if tp is not None else "—"

        if fill.get("net_pnl") is not None:
            net = float(fill.get("net_pnl") or 0.0)
            fees = float(fill.get("entry_fee_alloc") or 0.0) + float(fill.get("exit_fee") or 0.0)
            pnl_line = f"Net P&L: {net:+,.4f}$ | Fees: {fees:,.4f}$"
        else:
            pnl_line = f"P&L: {pnl:+,.2f}$ ({pnl_pct:+.2f}%, {pnl_r:+.2f}R)"

        text = (
            f"{result} CLOSED — {label}\n"
            f"{symbol} {side}\n"
            f"Entry {entry:,.4f} → Exit {exit_px:,.4f}\n"
            f"SL {sl_txt} | TP {tp_txt}\n"
            f"{pnl_line}\n"
            f"Reason: {reason}\n"
            f"Sentinel re-entry cooldown starts now."
        )

        loop = getattr(self, "_loop", None)
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(_send_close_with_retry(self, text), loop)
            return
        try:
            running = asyncio.get_event_loop()
            if running.is_running():
                running.create_task(_send_close_with_retry(self, text))
            else:
                running.run_until_complete(_send_close_with_retry(self, text))
        except Exception as exc:
            logger.error("[SENTINEL-CLOSE] unable to schedule EXIT alert: %s", exc)

    TelegramNotifier.notify_trade_closed = notify_trade_closed


def _install_reentry_cooldown() -> None:
    from .bot import TradingBot
    from .strategies.base import SignalType

    if getattr(TradingBot, "_sentinel_reentry_patch", False):
        return
    TradingBot._sentinel_reentry_patch = True

    original_closed = TradingBot._on_position_closed
    original_maybe = TradingBot._maybe_notify

    def _cooldown_seconds() -> float:
        try:
            mins = float(os.getenv("SENTINEL_REENTRY_COOLDOWN_MIN", "120"))
        except Exception:
            mins = 120.0
        return max(0.0, mins) * 60.0

    def _on_position_closed(self, symbol, strategy_name, exit_price, reason, strategy_inst=None):
        result = original_closed(self, symbol, strategy_name, exit_price, reason, strategy_inst)
        if str(strategy_name).lower().startswith("sentinel"):
            if not hasattr(self, "_sentinel_reentry_until"):
                self._sentinel_reentry_until = {}
            seconds = _cooldown_seconds()
            if seconds > 0:
                until = time.time() + seconds
                self._sentinel_reentry_until[symbol] = until
                logger.info(
                    "[SENTINEL-REENTRY] %s cooldown %.0f min after close (%s)",
                    symbol, seconds / 60.0, reason,
                )
        return result

    async def _maybe_notify(self, signal, sig_dict, strategy_name, candles=None):
        if (str(strategy_name).lower().startswith("sentinel")
                and signal.type != SignalType.HOLD):
            until_map = getattr(self, "_sentinel_reentry_until", {}) or {}
            until = float(until_map.get(signal.symbol, 0.0) or 0.0)
            remaining = until - time.time()
            if remaining > 0:
                logger.info(
                    "[SENTINEL-REENTRY] BLOCK %s %s — %.0f min cooldown remaining",
                    signal.symbol, signal.type.value.upper(), remaining / 60.0,
                )
                # V2/V3 may have tentatively set internal position state before
                # execution.  A cooldown veto must roll it back to avoid a
                # phantom position that would suppress future valid signals.
                try:
                    inst = self._resolve_strategy_inst(strategy_name)
                    if inst is not None and hasattr(inst, "cancel_pending_entry"):
                        inst.cancel_pending_entry("re-entry cooldown")
                except Exception:
                    pass
                return
            elif signal.symbol in until_map:
                until_map.pop(signal.symbol, None)

        return await original_maybe(self, signal, sig_dict, strategy_name, candles)

    TradingBot._on_position_closed = _on_position_closed
    TradingBot._maybe_notify = _maybe_notify


def install_sentinel_runtime_guard() -> None:
    _install_reliable_close_notifier()
    _install_reentry_cooldown()
    logger.info(
        "Sentinel runtime guard installed | reliable EXIT alerts | re-entry cooldown=%s min",
        os.getenv("SENTINEL_REENTRY_COOLDOWN_MIN", "120"),
    )
