"""
Persistent signal state — survives bot restarts.

Tracks:
  - Which symbols have an unresolved active signal (prevents re-entry)
  - Every signal fired (for signals/day stats)
  - Closed trade outcomes (win/loss history for /stats)
"""
import json
import logging
import os
import time
from collections import defaultdict

logger = logging.getLogger("signal_state")

_DEFAULT_PATH = os.environ.get("SIGNAL_STATE_FILE", "/app/signal_state.json")


class SignalState:

    def __init__(self, path: str = _DEFAULT_PATH):
        self.path = path
        self._active: dict[str, dict] = {}
        self._fired: list[dict] = []       # every signal alert sent
        self._outcomes: list[dict] = []    # closed trade results
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
            self._active   = data.get("active",   {})
            self._fired    = data.get("fired",     [])
            self._outcomes = data.get("outcomes",  [])
            logger.info("Signal state loaded: %d locks, %d fired, %d outcomes",
                        len(self._active), len(self._fired), len(self._outcomes))
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Could not load signal state: %s", e)

    def _save(self):
        try:
            with open(self.path, "w") as f:
                json.dump({
                    "active":   self._active,
                    "fired":    self._fired[-1000:],
                    "outcomes": self._outcomes[-500:],
                }, f, indent=2)
        except Exception as e:
            logger.warning("Could not save signal state: %s", e)

    # ------------------------------------------------------------------
    # Active signal lock
    # ------------------------------------------------------------------

    def is_locked(self, symbol: str) -> bool:
        return symbol in self._active

    def last_direction(self, symbol: str) -> tuple[str | None, int]:
        entry = self._active.get(symbol, {})
        return entry.get("direction"), entry.get("ts", 0)

    def lock(self, symbol: str, direction: str):
        self._active[symbol] = {"direction": direction, "ts": int(time.time() * 1000)}
        self._save()

    def unlock(self, symbol: str):
        if symbol in self._active:
            del self._active[symbol]
            self._save()

    # ------------------------------------------------------------------
    # Signal firing log
    # ------------------------------------------------------------------

    def record_signal(self, symbol: str, direction: str, price: float, confidence: float):
        """Called every time a signal alert is actually sent to Telegram."""
        self._fired.append({
            "symbol":     symbol,
            "direction":  direction,
            "price":      round(price, 4),
            "confidence": round(confidence, 2),
            "ts":         int(time.time() * 1000),
        })
        self._save()

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_outcome(self, symbol: str, side: str, entry: float,
                       exit_price: float, sl, tp, reason: str):
        risk  = abs(entry - sl) if sl else abs(entry - exit_price) or 1.0
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

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def signals_per_day(self) -> float:
        if not self._fired:
            return 0.0
        days: dict[int, int] = defaultdict(int)
        for s in self._fired:
            day_key = s["ts"] // (86_400_000)   # ms → day bucket
            days[day_key] += 1
        return round(sum(days.values()) / len(days), 1)

    def summary(self) -> dict:
        out = self._outcomes
        total_fired = len(self._fired)

        if not out:
            return {
                "trades": 0,
                "total_signals": total_fired,
                "signals_per_day": self.signals_per_day(),
            }

        wins       = [o for o in out if o["pnl_r"] > 0]
        losses     = [o for o in out if o["pnl_r"] <= 0]
        total_r    = sum(o["pnl_r"] for o in out)
        gross_win  = sum(o["pnl_r"] for o in wins)
        gross_loss = abs(sum(o["pnl_r"] for o in losses))
        pf         = round(gross_win / gross_loss, 2) if gross_loss else 999.0

        streak = 0
        if out:
            sign = 1 if out[-1]["pnl_r"] > 0 else -1
            for o in reversed(out):
                if (o["pnl_r"] > 0) == (sign == 1):
                    streak += sign
                else:
                    break

        return {
            "trades":          len(out),
            "wins":            len(wins),
            "losses":          len(losses),
            "win_rate":        round(len(wins) / len(out) * 100, 1),
            "profit_factor":   pf,
            "total_r":         round(total_r, 2),
            "streak":          streak,
            "total_signals":   total_fired,
            "signals_per_day": self.signals_per_day(),
            "recent":          out[-10:],
        }
