"""
Persistent signal state — survives bot restarts.

Tracks:
  - Which symbols have an unresolved active signal (prevents re-entry)
  - Every signal fired (for signals/day stats)
  - Closed trade outcomes (win/loss history for /stats)
  - Pending virtual trades (for SL/TP outcome tracking on forex / paper-0-balance signals)
"""
import json
import logging
import os
import time
from collections import defaultdict
from typing import Optional

logger = logging.getLogger("signal_state")

_DEFAULT_PATH = os.environ.get("SIGNAL_STATE_FILE", "/app/signal_state.json")
_PENDING_TTL_MS = 7 * 24 * 3600 * 1000  # 7 days


class SignalState:

    def __init__(self, path: str = _DEFAULT_PATH):
        self.path = path
        self._active: dict[str, dict] = {}
        self._fired: list[dict] = []       # every signal alert sent
        self._outcomes: list[dict] = []    # closed trade results
        self._pending: dict[str, dict] = {}  # virtual open trades awaiting SL/TP
        self._open_pos: dict[str, dict] = {}  # real positions — survive restarts
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
            self._active   = data.get("active",    {})
            self._fired    = data.get("fired",      [])
            self._outcomes = data.get("outcomes",   [])
            self._pending  = data.get("pending",    {})
            self._open_pos = data.get("open_pos",   {})
            logger.info("Signal state loaded: %d locks, %d fired, %d outcomes, %d pending, %d open_pos",
                        len(self._active), len(self._fired), len(self._outcomes),
                        len(self._pending), len(self._open_pos))
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
                    "pending":  self._pending,
                    "open_pos": self._open_pos,
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

    def is_locked_for_strategy(self, symbol: str, strategy: str) -> bool:
        return f"{symbol}||{strategy}" in self._active

    def lock_strategy(self, symbol: str, strategy: str, direction: str):
        self._active[f"{symbol}||{strategy}"] = {"direction": direction, "ts": int(time.time() * 1000)}
        self._save()

    def unlock_strategy(self, symbol: str, strategy: str):
        key = f"{symbol}||{strategy}"
        if key in self._active:
            del self._active[key]
            self._save()

    def count_active(self, symbol: str) -> int:
        return sum(1 for k in self._active if k.startswith(f"{symbol}||"))

    # ------------------------------------------------------------------
    # Open position persistence (survives restarts)
    # ------------------------------------------------------------------

    def save_position(self, key: str, symbol: str, side: str, entry: float,
                      amount: float, sl: Optional[float], tp: Optional[float],
                      strategy: str):
        """Persist a live open position so it survives bot restarts."""
        self._open_pos[key] = {
            "symbol":   symbol,
            "side":     side,
            "entry":    round(entry, 8),
            "amount":   amount,
            "sl":       sl,
            "tp":       tp,
            "strategy": strategy,
            "ts":       int(time.time() * 1000),
        }
        self._save()
        logger.info("Position persisted: %s [%s] entry=%.4f amount=%.6f", symbol, strategy, entry, amount)

    def remove_position(self, key: str):
        """Remove a persisted position (called when it closes)."""
        if key in self._open_pos:
            del self._open_pos[key]
            self._save()

    def get_saved_positions(self) -> list[dict]:
        """Return all persisted open positions (used on restart to restore state)."""
        return list(self._open_pos.values())

    # ------------------------------------------------------------------
    # Virtual outcome tracking (forex signal-only + paper/0-balance crypto)
    # ------------------------------------------------------------------

    def add_pending(self, key: str, symbol: str, side: str, entry: float,
                    sl: float, tp: float, strategy: str = ""):
        """Register a virtual open trade to monitor SL/TP virtually."""
        if not sl or not tp:
            return
        if key in self._pending:
            return
        self._pending[key] = {
            "symbol":   symbol,
            "side":     side,
            "entry":    round(entry, 8),
            "sl":       sl,
            "tp":       tp,
            "strategy": strategy,
            "ts":       int(time.time() * 1000),
        }
        self._save()
        logger.info("Virtual trade registered: %s %s @ %.4f  SL=%.4f TP=%.4f", side, symbol, entry, sl, tp)

    def check_and_resolve_pending(self, symbol: str, high: float, low: float) -> list[tuple[str, float]]:
        """
        Check all pending virtual trades for symbol against high/low prices.
        Resolves any that hit SL or TP: records the outcome and removes the entry.
        Returns list of (reason, exit_price) for each resolved trade.
        """
        resolved = []
        now = int(time.time() * 1000)
        changed = False
        for key in list(self._pending):
            item = self._pending[key]
            if item["symbol"] != symbol:
                continue
            # Prune expired entries
            if now - item["ts"] > _PENDING_TTL_MS:
                del self._pending[key]
                changed = True
                logger.info("Virtual trade expired (7d): %s", key)
                continue
            sl = item["sl"]; tp = item["tp"]
            side = item["side"]
            entry = item["entry"]
            hit = None; exit_price = None
            if side in ("buy", "long"):
                if low <= sl:
                    hit = "stop_loss";   exit_price = sl
                elif high >= tp:
                    hit = "take_profit"; exit_price = tp
            else:
                if high >= sl:
                    hit = "stop_loss";   exit_price = sl
                elif low <= tp:
                    hit = "take_profit"; exit_price = tp
            if hit:
                del self._pending[key]
                changed = True
                self.record_outcome(
                    symbol=symbol, side=side,
                    entry=entry, exit_price=exit_price,
                    sl=sl, tp=tp, reason=hit,
                    strategy=item.get("strategy", ""),
                )
                resolved.append((hit, exit_price))
                logger.info("Virtual %s %s → %s @ %.4f (entry %.4f)", side, symbol, hit, exit_price, entry)
        if changed and not resolved:  # record_outcome already saves when resolved
            self._save()
        return resolved

    # ------------------------------------------------------------------
    # Signal firing log
    # ------------------------------------------------------------------

    def record_signal(self, symbol: str, direction: str, price: float,
                      confidence: float, strategy: str = ""):
        """Called every time a signal alert is actually sent to Telegram."""
        self._fired.append({
            "symbol":     symbol,
            "direction":  direction,
            "price":      round(price, 4),
            "confidence": round(confidence, 2),
            "strategy":   strategy,
            "ts":         int(time.time() * 1000),
        })
        self._save()

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_outcome(self, symbol: str, side: str, entry: float,
                       exit_price: float, sl, tp, reason: str, strategy: str = ""):
        risk  = abs(entry - sl) if sl else abs(entry - exit_price) or 1.0
        pnl_r = abs(exit_price - entry) / risk if reason == "take_profit" else -1.0
        self._outcomes.append({
            "symbol":   symbol,
            "side":     side,
            "entry":    round(entry, 4),
            "exit":     round(exit_price, 4),
            "sl":       sl,
            "tp":       tp,
            "pnl_r":    round(pnl_r, 2),
            "reason":   reason,
            "strategy": strategy,
            "ts":       int(time.time() * 1000),
        })
        self._save()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def strategy_stats(self, days: int = 7) -> dict:
        """Per-strategy signal count and WR for the past N days."""
        cutoff = int(time.time() * 1000) - days * 86_400_000
        fired_r    = [f for f in self._fired    if f["ts"] >= cutoff]
        outcomes_r = [o for o in self._outcomes if o["ts"] >= cutoff]

        data: dict[str, dict] = {}
        for f in fired_r:
            s = f.get("strategy") or "unknown"
            if s not in data:
                data[s] = {"signals": 0, "wins": 0, "losses": 0}
            data[s]["signals"] += 1
        for o in outcomes_r:
            s = o.get("strategy") or "unknown"
            if s not in data:
                data[s] = {"signals": 0, "wins": 0, "losses": 0}
            if o["pnl_r"] > 0:
                data[s]["wins"]   += 1
            else:
                data[s]["losses"] += 1

        result = {}
        for s, d in sorted(data.items()):
            closed = d["wins"] + d["losses"]
            wr = round(d["wins"] / closed * 100, 1) if closed else None
            result[s] = {
                "signals": d["signals"],
                "wins":    d["wins"],
                "losses":  d["losses"],
                "win_rate": wr,
            }
        return result

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
                "pending": len(self._pending),
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
            "trades":             len(out),
            "wins":               len(wins),
            "losses":             len(losses),
            "win_rate":           round(len(wins) / len(out) * 100, 1),
            "profit_factor":      pf,
            "total_r":            round(total_r, 2),
            "streak":             streak,
            "pending":            len(self._pending),
            "total_signals":      total_fired,
            "signals_per_day":    self.signals_per_day(),
            "strategy_breakdown": self.strategy_stats(days=7),
            "recent":             out[-10:],
        }
