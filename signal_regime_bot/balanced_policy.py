"""V3.3.1 Balanced adaptive policy.

Designed as a light policy layer:
- SpikeGuard remains disabled by default in config.
- XAU is disabled by default.
- Lifecycle penalties are intentionally small.
- Re-entry penalty is capped; there is no mandatory fresh-BOS hard gate.
"""

import os

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

def symbol_enabled(symbol: str) -> bool:
    s = (symbol or "").upper()
    if s.startswith("XAU"):
        return _env_bool("XAU_TRADING_ENABLED", False)
    return True

def lifecycle_penalty(stage: str) -> int:
    return {
        "EARLY": 0,
        "DEVELOPING": 0,
        "MATURE": 1,
        "EXTENDED": 3,
        "EXHAUSTING": 5,
    }.get((stage or "").upper(), 0)

def same_leg_penalty(entries_in_leg: int) -> int:
    if entries_in_leg <= 1:
        return 0
    if entries_in_leg == 2:
        return 2
    return 4  # capped; intentionally no +8 / fresh-BOS hard gate

def adjusted_threshold(base_threshold: int, stage: str, entries_in_leg: int) -> int:
    return int(base_threshold + lifecycle_penalty(stage) + same_leg_penalty(entries_in_leg))
