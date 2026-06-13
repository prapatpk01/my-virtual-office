"""
Persistent signal state — survives bot restarts.

Tracks:
  - Which symbols have an unresolved active signal (prevents re-entry)
  - Closed trade outcomes (win/loss history for /stats)

Written to a JSON file on every change so Railway restarts don't
lose the "already in a trade" guard.
"""
import json
import logging
import os
import time

logger = logging.getLogger("signal_state")

_DEFAULT_PATH = os.environ.get("SIGNAL_STATE_FILE", "/app/signal_state.json")


class SignalState:

    def __init__(self, path: str = _DEFAULT_PATH):
        self.path = path
        self._active: dict[str, dict] = {}   # symbol → {direction, ts}
        self._outcomes: list[dict] = []       # all closed trades
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
            self._active   = data.get("active", {})
            self._outcomes = data.get("outcomes", [])
            logger.info("Signal state loaded: %d active locks, %d outcomes",
                        len(self._active), len(self._outcomes))
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Could not load signal state: %s", e)

    def _save(self):
        try:
            with open(self.path, "w") as f:
                json.dump({
                    "active":   self._active,
                    "outcomes": self._outcomes[-500:],   # keep last 500
                }, f, indent=2)
        except Exception as e:
            logger.warning("Could not save signal state: %s", e)

    # ------------------------------------------------------------------
    # Active signal lock (one per symbol)
    # ------------------------------------------------------------------

    def is_locked(self, symbol: str) -> bool:
        """True if symbol already has an unresolved active signal."""
        return symbol in self._active

    def last_direction(self, symbol: str) -> tuple[str | None, int]:
        entry = self._active.get(symbol, {})
        return entry.get("direction"), entry.get("ts", 0)

    def lock(self, symbol: str, direction: str):
        """Call when a new signal fires — blocks further signals for symbol."""
        self._active[symbol] = {
            "direction": direction,
            "ts": int(time.time() * 1000),
        }
        self._save()

    def unlock(self, symbol: str):
        """Call when SL or TP is hit — allows next signal for symbol."""
        if symbol in self._active:
            del self._active[symbol]
            self._save()

    # ------------------------------------------------------------------
    # Outcome recording & statistics
    # ------------------------------------------------------------------

    def record_outcome(self, symbol: str, side: str, entry: float,
                       exit_price: float, sl, tp, reason: str):
        risk = abs(entry - sl) if sl else abs(entry - exit_price) or 1.0
        pnl_r = abs(exit_price - entry) / risk if reason == "take_profit" else -1.0
        self._outcomes.append({
            "symbol": symbol,
            "side":   side,
            "entry":  round(entry, 4),
            "exit":   round(exit_price, 4),
            "sl":     sl,
            "tp":     tp,
            "pnl_r":  round(pnl_r, 2),
            "reason": reason,
            "ts":     int(time.time() * 1000),
        })
        self._save()

    def summary(self) -> dict:
        out = self._outcomes
        if not out:
            return {"trades": 0}
        wins        = [o for o in out if o["pnl_r"] > 0]
        losses      = [o for o in out if o["pnl_r"] <= 0]
        total_r     = sum(o["pnl_r"] for o in out)
        gross_win   = sum(o["pnl_r"] for o in wins)
        gross_loss  = abs(sum(o["pnl_r"] for o in losses))
        pf          = round(gross_win / gross_loss, 2) if gross_loss else 999.0
        streak = 0
        if out:
            sign = 1 if out[-1]["pnl_r"] > 0 else -1
            for o in reversed(out):
                if (o["pnl_r"] > 0) == (sign == 1):
                    streak += sign
                else:
                    break
        return {
            "trades":         len(out),
            "wins":           len(wins),
            "losses":         len(losses),
            "win_rate":       round(len(wins) / len(out) * 100, 1),
            "profit_factor":  pf,
            "total_r":        round(total_r, 2),
            "streak":         streak,
            "recent":         out[-10:],
        }
