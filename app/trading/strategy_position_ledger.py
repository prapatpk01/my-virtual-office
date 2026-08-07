"""Persistent per-strategy ownership ledger for aggregated exchange positions.

OKX hedge mode separates LONG and SHORT, but same-symbol same-side entries from
multiple strategies are aggregated into one exchange position.  The bot still
needs to remember which strategy owns which slice (entry, amount, side) so each
strategy can close only its own amount and recover that ownership after restart.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Optional

logger = logging.getLogger("strategy_position_ledger")


def _default_path() -> str:
    signal_path = os.getenv("SIGNAL_STATE_FILE", "/app/signal_state.json")
    base_dir = os.path.dirname(signal_path) or "/app"
    return os.getenv(
        "STRATEGY_POSITION_LEDGER_FILE",
        os.path.join(base_dir, "strategy_position_ledger.json"),
    )


class StrategyPositionLedger:
    def __init__(self, path: Optional[str] = None):
        self.path = path or _default_path()
        self._positions: dict[str, dict] = {}
        self._load()

    @staticmethod
    def key(symbol: str, strategy: str) -> str:
        return f"{symbol}||{strategy}"

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            raw = payload.get("positions", payload)
            self._positions = raw if isinstance(raw, dict) else {}
            logger.info(
                "Strategy ownership ledger loaded: %d position slice(s)",
                len(self._positions),
            )
        except FileNotFoundError:
            self._positions = {}
        except Exception as exc:
            logger.warning("Could not load strategy ownership ledger: %s", exc)
            self._positions = {}

    def _save(self) -> None:
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        payload = {
            "updated_at": int(time.time() * 1000),
            "positions": self._positions,
        }
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix="strategy_position_ledger_",
                suffix=".json",
                dir=directory,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        except Exception as exc:
            logger.warning("Could not save strategy ownership ledger: %s", exc)
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def set(
        self,
        symbol: str,
        strategy: str,
        side: str,
        entry: float,
        amount: float,
        stop_loss=None,
        take_profit=None,
    ) -> None:
        if not symbol or not strategy or amount <= 0:
            return
        self._positions[self.key(symbol, strategy)] = {
            "symbol": symbol,
            "strategy": strategy,
            "side": str(side).lower(),
            "entry": float(entry),
            "amount": float(amount),
            "stop_loss": float(stop_loss) if stop_loss else None,
            "take_profit": float(take_profit) if take_profit else None,
            "updated_at": int(time.time() * 1000),
        }
        self._save()

    def remove(self, symbol: str, strategy: str) -> Optional[dict]:
        removed = self._positions.pop(self.key(symbol, strategy), None)
        if removed is not None:
            self._save()
        return removed

    def get(self, symbol: str, strategy: str) -> Optional[dict]:
        item = self._positions.get(self.key(symbol, strategy))
        return dict(item) if isinstance(item, dict) else None

    def all_for_symbol(self, symbol: str) -> list[dict]:
        return [
            dict(value)
            for value in self._positions.values()
            if isinstance(value, dict) and value.get("symbol") == symbol
        ]

    def all(self) -> list[dict]:
        return [dict(value) for value in self._positions.values() if isinstance(value, dict)]

    def update_amount(self, symbol: str, strategy: str, amount: float) -> None:
        key = self.key(symbol, strategy)
        item = self._positions.get(key)
        if not isinstance(item, dict):
            return
        if amount <= 1e-12:
            self._positions.pop(key, None)
        else:
            item["amount"] = float(amount)
            item["updated_at"] = int(time.time() * 1000)
        self._save()
