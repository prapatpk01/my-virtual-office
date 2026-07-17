"""Diagnostics (spec §31, §33) — reason-code accounting + view log lines.

Separates hard rejects / quality reductions / risk reductions / info.
"""
from __future__ import annotations

import logging
import time
from collections import Counter, deque
from typing import Optional

logger = logging.getLogger("dual_entry.diag")

_HARD_PREFIXES = ("REJECT_",)


class DiagnosticEngine:
    def __init__(self, window: int = 100):
        self.window = window
        self.rejections: deque = deque(maxlen=1000)
        self.errors: deque = deque(maxlen=200)
        self.counters: Counter = Counter()
        self.ops: Counter = Counter()        # operational window counts
        self.view: dict = {}                 # symbol -> latest view line

    def record_rejection(self, symbol: str, reason_codes: list) -> None:
        ts = int(time.time() * 1000)
        for code in reason_codes:
            kind = "HARD" if any(code.startswith(p) for p in _HARD_PREFIXES) else "INFO"
            self.rejections.append({"ts": ts, "symbol": symbol, "code": code, "kind": kind})
            self.counters[code.split(":")[0]] += 1
        self.ops["rejected_signals"] += 1

    def record_error(self, symbol: str, error: Exception) -> None:
        self.errors.append({"ts": int(time.time() * 1000), "symbol": symbol,
                            "error": str(error)[:400]})
        self.ops["data_errors"] += 1
        logger.error("[%s] %s", symbol, error, exc_info=True)

    def log_cooldown(self, symbol: str, state) -> None:
        self.view[symbol] = f"{_sym(symbol)} | COOLDOWN | Resume=Next Candle"

    def count(self, key: str, n: int = 1) -> None:
        self.ops[key] += n

    # ── view log (spec §33) ──────────────────────────────────────────────────

    def set_view(self, symbol: str, line: str) -> None:
        self.view[symbol] = line

    def view_lines(self) -> list:
        return [self.view[s] for s in sorted(self.view)]

    def top_reasons(self, n: int = 8) -> list:
        return self.counters.most_common(n)


def _sym(symbol: str) -> str:
    return symbol.split("/")[0]
