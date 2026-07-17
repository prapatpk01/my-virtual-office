"""Atomic per-symbol state persistence + execution journal (spec §4).

tmp-file + os.replace = atomic on POSIX. state_version increments on every
save (optimistic locking guard); the journal records every order INTENT
before the order is sent, so a crash between intent and ack is detectable.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from .models import SymbolState

logger = logging.getLogger("dual_entry.state")


class StateStore:
    def __init__(self, state_dir: str):
        self.dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self._cache: dict = {}

    def _path(self, symbol: str) -> str:
        safe = symbol.replace("/", "_").replace(":", "_")
        return os.path.join(self.dir, f"{safe}.json")

    def get(self, symbol: str) -> SymbolState:
        if symbol in self._cache:
            return self._cache[symbol]
        p = self._path(symbol)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    st = SymbolState.from_dict(json.load(f))
                self._cache[symbol] = st
                return st
            except Exception as e:
                logger.error("[STATE] load failed %s: %s — starting fresh", symbol, e)
        st = SymbolState(symbol=symbol)
        self._cache[symbol] = st
        return st

    def save_atomic(self, symbol: str, state: SymbolState) -> None:
        state.state_version += 1
        p = self._path(symbol)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state.to_dict(), f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        self._cache[symbol] = state

    # ── execution journal ────────────────────────────────────────────────────

    def journal(self, symbol: str, event: str, payload: dict) -> None:
        p = os.path.join(self.dir, "execution_journal.jsonl")
        rec = {"ts": int(time.time() * 1000), "symbol": symbol, "event": event, **payload}
        with open(p, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def journal_has_intent(self, client_order_id: str) -> bool:
        p = os.path.join(self.dir, "execution_journal.jsonl")
        if not os.path.exists(p):
            return False
        try:
            with open(p) as f:
                for line in f:
                    if client_order_id in line and '"ORDER_INTENT"' in line:
                        return True
        except Exception:
            return False
        return False
