"""Guaranteed entry-alert delivery + persistent PAPER state for EMA Hybrid A-E.

The inherited HMA entry path persists a newly opened position before rendering /
sending its Telegram chart. That is correct for trade safety, but a downstream
chart or Telegram failure can leave a real/open PAPER position with no visible
entry notification. This mixin makes notification delivery observable and
recovers it without changing entry, risk, TP or SL logic.

It also keeps PAPER balance/positions on disk and rewrites /stats so A-E are
always reported separately. With STATE_DIR on a Railway persistent volume,
balance, positions, local state and the EMA journal survive redeploys.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone

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
    """Entry-alert guarantee + PAPER persistence + unified A-E statistics."""

    ENTRY_ALERT_DELIVERY_WINDOW_SEC = float(
        os.getenv("EMA_ENTRY_ALERT_DELIVERY_WINDOW_SEC", "120")
    )
    ENTRY_ALERT_RECOVER_EXISTING_E = _env_bool(
        "EMA_ENTRY_ALERT_RECOVER_EXISTING_E", True
    )
    ENTRY_ALERT_EXISTING_MAX_AGE_SEC = float(
        os.getenv("EMA_ENTRY_ALERT_EXISTING_MAX_AGE_SEC", str(6 * 60 * 60))
    )
    PAPER_START_BALANCE = float(os.getenv("EMA_PAPER_START_BALANCE", "10000"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ema_entry_delivery: dict[tuple[str, str], float] = {}
        self._install_entry_delivery_guard()
        self._install_paper_persistence()

    @staticmethod
    def _root(symbol: str) -> str:
        return str(symbol or "").split("/")[0].upper()

    # ------------------------------------------------------------------
    # PAPER account persistence
    # ------------------------------------------------------------------
    def _paper_account_file(self) -> str:
        return os.path.join(self.cfg.state_dir, "ema_hybrid_paper_account.json")

    def _paper_migration_snapshot(self) -> tuple[float, dict]:
        """Best-effort one-time reconstruction when no account file exists yet."""
        balance = self.PAPER_START_BALANCE
        try:
            balance += sum(float(r.get("pnl") or 0.0) for r in getattr(self, "ema_journal", []))
        except Exception:
            pass

        positions: dict[str, dict] = {}
        fee_rate = float(getattr(self.cfg, "fee_rate", 0.0) or 0.0)
        for symbol in getattr(self.cfg, "symbols", []):
            pos = ((getattr(self, "state", {}) or {}).get(symbol) or {}).get("pos") or {}
            side = str(pos.get("side") or "").lower()
            entry = float(pos.get("entry") or 0.0)
            amount = float(pos.get("amount") or 0.0)
            if side not in {"long", "short"} or entry <= 0 or amount <= 0:
                continue
            positions[f"{symbol}||{side}"] = {
                "entry": entry,
                "amount": amount,
                "tp": float(pos.get("tp") or 0.0) or None,
                "sl": float(pos.get("sl") or 0.0) or None,
            }
            initial_amount = float(pos.get("initial_amount") or amount)
            balance -= initial_amount * entry * fee_rate
            balance += float(pos.get("tp1_net_pnl") or 0.0)
        return balance, positions

    def _save_paper_account(self) -> None:
        if not bool(getattr(self.cfg, "paper", False)):
            return
        path = self._paper_account_file()
        try:
            os.makedirs(self.cfg.state_dir, exist_ok=True)
            payload = {
                "version": 1,
                "updated_ms": int(time.time() * 1000),
                "balance": dict(getattr(self.client, "_paper_balance", {}) or {}),
                "positions": dict(getattr(self.client, "_paper_positions", {}) or {}),
            }
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, separators=(",", ":"))
            os.replace(tmp, path)
        except Exception as exc:
            self._entry_alert_log("error", "PAPER account save failed: %s", exc, exc_info=True)

    def _install_paper_persistence(self) -> None:
        if not bool(getattr(self.cfg, "paper", False)):
            return

        path = self._paper_account_file()
        loaded = False
        try:
            with open(path) as f:
                payload = json.load(f)
            balance = payload.get("balance") or {}
            positions = payload.get("positions") or {}
            if isinstance(balance, dict) and isinstance(positions, dict):
                self.client._paper_balance = {
                    "USDT": float(balance.get("USDT", self.PAPER_START_BALANCE))
                }
                self.client._paper_positions = positions
                loaded = True
                self._entry_alert_log(
                    "info", "loaded persistent PAPER account balance=%.2f positions=%d from %s",
                    self.client._paper_balance["USDT"], len(positions), path,
                )
        except FileNotFoundError:
            pass
        except Exception as exc:
            self._entry_alert_log(
                "error", "PAPER account load failed; using migration snapshot: %s", exc,
                exc_info=True,
            )

        if not loaded:
            balance, positions = self._paper_migration_snapshot()
            self.client._paper_balance = {"USDT": float(balance)}
            self.client._paper_positions = positions
            self._entry_alert_log(
                "warning",
                "created persistent PAPER account from local state balance=%.2f positions=%d",
                balance, len(positions),
            )

        original_paper_order = getattr(self.client, "_paper_order", None)
        if callable(original_paper_order):
            async def persistent_paper_order(*args, __fn=original_paper_order, **kwargs):
                result = __fn(*args, **kwargs)
                if hasattr(result, "__await__"):
                    result = await result
                self._save_paper_account()
                return result
            self.client._paper_order = persistent_paper_order

        self._save_paper_account()

    # ------------------------------------------------------------------
    # Unified A-E stats
    # ------------------------------------------------------------------
    @staticmethod
    def _row_setup_key(row: dict) -> str:
        stored = str(row.get("setup") or "").upper()
        if stored in {
            "A_EMA_CROSS", "B_PULLBACK_RECLAIM", "C_BOLL_MACD_KDJ",
            "D_MA5_MA20", "E_MACD_VOLUME",
        }:
            return stored
        return _setup_key(str(row.get("trigger") or ""))

    @staticmethod
    def _setup_metrics(rows: list[dict]) -> tuple[int, float, str, float]:
        total = len(rows)
        if total == 0:
            return 0, 0.0, "—", 0.0
        wins = sum(1 for r in rows if float(r.get("pnl") or 0.0) > 0)
        gross_win = sum(float(r.get("pnl") or 0.0) for r in rows if float(r.get("pnl") or 0.0) > 0)
        gross_loss = abs(sum(float(r.get("pnl") or 0.0) for r in rows if float(r.get("pnl") or 0.0) < 0))
        net = sum(float(r.get("pnl") or 0.0) for r in rows)
        wr = wins / total * 100.0
        if gross_loss > 1e-12:
            pf_text = f"{gross_win / gross_loss:.2f}"
        elif gross_win > 0:
            pf_text = "∞"
        else:
            pf_text = "0.00"
        return total, wr, pf_text, net

    async def _build_stats_report(self) -> str:
        report = await super()._build_stats_report()
        if not bool(getattr(self.cfg, "paper", False)):
            return report

        now = datetime.now(timezone.utc)
        month_start = int(datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp() * 1000)
        month_rows = [
            r for r in getattr(self, "ema_journal", [])
            if int(r.get("close_ms") or 0) >= month_start
        ]
        specs = (
            ("A_EMA_CROSS", "A EMA CROSS"),
            ("B_PULLBACK_RECLAIM", "B PULLBACK"),
            ("C_BOLL_MACD_KDJ", "C BOLL/MACD/KDJ"),
            ("D_MA5_MA20", "D MA5/MA20"),
            ("E_MACD_VOLUME", "E MACD/VOLUME"),
        )

        setup_lines: list[str] = []
        for key, label in specs:
            block = [r for r in month_rows if self._row_setup_key(r) == key]
            total, wr, pf_text, net = self._setup_metrics(block)
            if total:
                setup_lines.append(
                    f"{label:17s} {total} | {wr:.0f}% WR | PF {pf_text} | ${net:+.2f}"
                )
            else:
                setup_lines.append(f"{label:17s} 0 | — WR | PF — | $+0.00")

        lines = report.splitlines()
        section = next((i for i, line in enumerate(lines) if line.startswith("BY SETUP —")), None)
        since = next(
            (i for i, line in enumerate(lines) if line.startswith("SINCE ") and (section is None or i > section)),
            None,
        )
        if section is None or since is None:
            return report

        sep = "――――――――――――――――"
        start = section - 1 if section > 0 and lines[section - 1] == sep else section
        end = since - 1 if since > 0 and lines[since - 1] == sep else since
        block = [
            sep,
            f"BY SETUP — {now.strftime('%b %Y')}",
            sep,
            *setup_lines,
            "",
        ]
        return "\n".join(lines[:start] + block + lines[end:])

    # ------------------------------------------------------------------
    # Entry alert guarantee
    # ------------------------------------------------------------------
    def _entry_identity_from_text(self, text: str):
        if not isinstance(text, str):
            return None
        plain = text.replace("`", "").replace("*", "")
        upper = plain.upper()
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
                    "warning", "entry photo delivery returned false; forcing text fallback",
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
                    symbol, df5, side, entry, sl, tp, tp,
                    ema_fast_len=8, ema_slow_len=13, tf_label="5M",
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
        before_open = bool(st.get("pos"))
        try:
            result = await super()._look_for_entry(symbol, st)
        except Exception as exc:
            pos = st.get("pos") or {}
            if not before_open and pos:
                self._entry_alert_log(
                    "error", "post-order entry pipeline failed for %s; recovering alert: %s",
                    symbol, exc, exc_info=True,
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
                    "warning", "position opened but no confirmed entry delivery: %s setup=%s trigger=%s",
                    symbol, _setup_key(str(pos.get("trigger") or "")), pos.get("trigger"),
                )
                await self._send_entry_recovery_alert(symbol, pos)
            if changed:
                self._save_state()
        return result

    async def _manage(self, symbol: str, st: dict):
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
