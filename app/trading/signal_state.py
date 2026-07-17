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

logger = logging.getLogger("signal_state")

_DEFAULT_PATH = os.environ.get("SIGNAL_STATE_FILE", "/app/signal_state.json")
_PENDING_TTL_MS = 7 * 24 * 3600 * 1000  # 7 days

# ── Paper-account model (mirrors how the live bot is asked to size) ──────
#   start $1000, each trade uses 5% of the *current* balance as margin,
#   opened at 20x leverage. Every closed trade books a real $ P&L into this
#   account so notifications show money, not abstract R units.
PAPER_START_BALANCE = float(os.environ.get("PAPER_START_BALANCE", "1000"))
PAPER_MARGIN_PCT    = float(os.environ.get("PAPER_MARGIN_PCT", "0.05"))
PAPER_LEVERAGE      = float(os.environ.get("PAPER_LEVERAGE", "20"))
PAPER_TAKER_FEE     = float(os.environ.get("PAPER_TAKER_FEE", "0.0005"))  # per side


def classify_exit_reason(reason: str, won: bool) -> tuple[str, str]:
    """Map a raw exit-reason string to a human label + emoji.

    The strategy emits many exit reasons (hard SL/TP, EMA cross-back,
    trailed break-even stop, TP1 partial). The old code labelled anything
    that wasn't literally ``take_profit`` as a "Stop-Loss Hit", which was
    wrong for trend-exit and break-even closes. This inspects the reason
    text and returns what actually happened.
    """
    r = (reason or "").lower()
    if "partial" in r or "tp1" in r:
        return ("Partial Take-Profit", "💰")
    if "take_profit" in r or "tp2" in r or "hard_tp" in r or "take profit" in r:
        return ("Take-Profit Hit", "🎯")
    if "be+" in r or "break" in r or "trailed" in r or "trail" in r:
        return ("Break-Even / Trailing Stop", "🟰")
    if "stop_loss" in r or "hard_sl" in r or "stop loss" in r:
        return ("Stop-Loss Hit", "🛑")
    if "exit" in r or "hma" in r or "cross" in r or "trend" in r:
        return ("Trend Exit (EMA cross-back)", "↩️")
    return ("Position Closed", "☑️")


class SignalState:

    def __init__(self, path: str = _DEFAULT_PATH):
        self.path = path
        self._active: dict[str, dict] = {}
        self._fired: list[dict] = []       # every signal alert sent
        self._outcomes: list[dict] = []    # closed trade results
        self._pending: dict[str, dict] = {}  # virtual open trades awaiting SL/TP
        self._paper_balance: float = PAPER_START_BALANCE  # running paper account
        self._paper_positions: dict[str, dict] = {}  # symbol||strategy -> open paper pos
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
            self._pending  = data.get("pending",   {})
            self._paper_balance = float(data.get("paper_balance", PAPER_START_BALANCE))
            self._paper_positions = data.get("paper_positions", {})
            logger.info("Signal state loaded: %d locks, %d fired, %d outcomes, %d pending, paper $%.2f",
                        len(self._active), len(self._fired), len(self._outcomes),
                        len(self._pending), self._paper_balance)
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
                    "paper_balance": round(self._paper_balance, 2),
                    "paper_positions": self._paper_positions,
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

    def check_and_resolve_pending(self, symbol: str, high: float, low: float) -> list[tuple[str, float, dict]]:
        """
        Check all pending virtual trades for symbol against high/low prices.
        Resolves any that hit SL or TP: records the outcome and removes the entry.
        Returns list of (reason, exit_price, outcome) for each resolved trade.
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
                outcome = self.record_outcome(
                    symbol=symbol, side=side,
                    entry=entry, exit_price=exit_price,
                    sl=sl, tp=tp, reason=hit,
                    strategy=item.get("strategy", ""),
                )
                resolved.append((hit, exit_price, outcome))
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

    @property
    def paper_balance(self) -> float:
        return self._paper_balance

    def _paper_key(self, symbol: str, strategy: str) -> str:
        return f"{symbol}||{strategy}"

    def open_paper_position(self, symbol: str, side: str, entry: float,
                            sl, strategy: str = "") -> None:
        """Snapshot a paper position at entry so partial TPs and the final
        close book consistently against one fixed size (5% margin × 20x of the
        balance at open), instead of re-sizing off the balance at close time."""
        if not entry:
            return
        margin   = self._paper_balance * PAPER_MARGIN_PCT
        notional = margin * PAPER_LEVERAGE
        amount   = notional / entry
        self._paper_positions[self._paper_key(symbol, strategy)] = {
            "side":       side,
            "entry":      entry,
            "sl":         sl,
            "amount":     amount,       # remaining size (shrinks on partials)
            "init_amount": amount,      # original size (for R computation)
            "risk_price": abs(entry - sl) if sl else 0.0,
            "realized":   0.0,          # net $ already banked from partials
            "ts":         int(time.time() * 1000),
        }
        self._save()

    def record_paper_partial(self, symbol: str, exit_price: float,
                             close_frac: float, strategy: str = "") -> dict | None:
        """Book a partial close (e.g. TP1 taking 50%) into the paper account.
        Returns {pnl_usd, pnl_pct, balance_after} for the notifier, or None if
        no paper position is tracked (e.g. reconciled/legacy positions)."""
        pp = self._paper_positions.get(self._paper_key(symbol, strategy))
        if not pp or pp["amount"] <= 0:
            return None
        is_long = pp["side"] in ("buy", "long")
        amt   = pp["amount"] * close_frac
        entry = pp["entry"]
        gross = (exit_price - entry) * amt if is_long else (entry - exit_price) * amt
        fees  = (entry + exit_price) * amt * PAPER_TAKER_FEE
        pnl   = gross - fees
        base  = self._paper_balance
        self._paper_balance += pnl
        pp["realized"] += pnl
        pp["amount"]   -= amt
        self._save()
        return {
            "pnl_usd": round(pnl, 2),
            "pnl_pct": round((pnl / base * 100) if base else 0.0, 2),
            "balance_after": round(self._paper_balance, 2),
        }

    def record_outcome(self, symbol: str, side: str, entry: float,
                       exit_price: float, sl, tp, reason: str, strategy: str = "",
                       fill: dict | None = None) -> dict:
        """Book a closed trade into the paper account and log the outcome.

        Computes the *actual* directional P&L (in $ and %) from a $1000
        paper account sized at 5% margin × 20x leverage — for EVERY exit
        reason, not just take-profits. If a paper position was snapshotted at
        entry (open_paper_position), the remaining size is closed at `exit`
        and any banked partial (`realized`) is folded into the trade's total
        so the balance stays coherent. Returns the outcome dict.
        """
        is_long = side in ("buy", "long")
        pp = self._paper_positions.pop(self._paper_key(symbol, strategy), None)

        if pp is not None:
            # Coherent close: use the snapshotted size + any banked partials.
            entry_px = pp["entry"]
            amt      = pp["amount"]
            gross    = (exit_price - entry_px) * amt if is_long else (entry_px - exit_price) * amt
            fees     = (entry_px + exit_price) * amt * PAPER_TAKER_FEE
            leg_pnl  = gross - fees
            pnl_usd  = pp["realized"] + leg_pnl
            self._paper_balance += leg_pnl
            init_risk = pp["init_amount"] * pp["risk_price"]
            pnl_r    = (pnl_usd / init_risk) if init_risk > 0 else 0.0
        else:
            # Fallback (reconciled/legacy/virtual): size off the balance now.
            price_move = (exit_price - entry) if is_long else (entry - exit_price)
            risk = abs(entry - sl) if sl else (abs(entry - exit_price) or 1.0)
            pnl_r = price_move / risk if risk else 0.0
            margin   = self._paper_balance * PAPER_MARGIN_PCT
            notional = margin * PAPER_LEVERAGE
            amount   = (notional / entry) if entry else 0.0
            gross    = price_move * amount
            fees     = (entry + exit_price) * amount * PAPER_TAKER_FEE
            pnl_usd  = gross - fees
            self._paper_balance += pnl_usd

        base_bal = self._paper_balance - pnl_usd
        pnl_pct  = (pnl_usd / base_bal * 100) if base_bal else 0.0

        # When the exchange's post-fill accounting is available (avgPx/fillSz/
        # fee/realized pnl -> net_pnl), ITS sign decides win/loss — the real
        # fill is the truth; the paper model above is just the running $1000
        # simulation for stats.
        won = (fill["net_pnl"] > 0) if (fill and fill.get("net_pnl") is not None) else pnl_usd > 0
        label, emoji = classify_exit_reason(reason, won)

        outcome = {
            "symbol":       symbol,
            "side":         side,
            "entry":        round(entry, 4),
            "exit":         round(exit_price, 4),
            "sl":           sl,
            "tp":           tp,
            "pnl_r":        round(pnl_r, 2),
            "pnl_usd":      round(pnl_usd, 2),
            "pnl_pct":      round(pnl_pct, 2),
            "balance_after": round(self._paper_balance, 2),
            "reason":       reason,
            "reason_label": label,
            "emoji":        emoji,
            "won":          won,
            "strategy":     strategy,
            "fill":         fill,
            "ts":           int(time.time() * 1000),
        }
        self._outcomes.append(outcome)
        self._save()
        return outcome

    # ------------------------------------------------------------------
    # Deep-dive learning analysis
    # ------------------------------------------------------------------

    def deep_analysis(self, days: int = 30) -> dict:
        """
        Runs LearningAnalysis over the full fired/outcome history and returns
        deep insights (win-rate by confidence/strategy/symbol/hour, trend,
        and plain-language recommendations).
        """
        from .learning_analysis import LearningAnalysis
        return LearningAnalysis(self._fired, self._outcomes).analyze(days=days)

    def ai_context(self, days: int = 30) -> str:
        """Short text digest of historical performance for use in AI prompts."""
        from .learning_analysis import LearningAnalysis
        return LearningAnalysis(self._fired, self._outcomes).context_for_ai(days=days)

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
            if o.get("won", o.get("pnl_r", 0) > 0):
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

        def _won(o):
            return o.get("won", o.get("pnl_r", 0) > 0)

        if not out:
            return {
                "trades": 0,
                "pending": len(self._pending),
                "total_signals": total_fired,
                "signals_per_day": self.signals_per_day(),
                "start_balance": round(PAPER_START_BALANCE, 2),
                "paper_balance": round(self._paper_balance, 2),
                "total_pnl_usd": 0.0,
                "return_pct": round((self._paper_balance / PAPER_START_BALANCE - 1) * 100, 2)
                              if PAPER_START_BALANCE else 0.0,
            }

        wins       = [o for o in out if _won(o)]
        losses     = [o for o in out if not _won(o)]
        total_r    = sum(o.get("pnl_r", 0) for o in out)
        # Profit factor on real money (gross $ won / gross $ lost).
        gross_win_usd  = sum(o.get("pnl_usd", 0) for o in wins)
        gross_loss_usd = abs(sum(o.get("pnl_usd", 0) for o in losses))
        pf         = round(gross_win_usd / gross_loss_usd, 2) if gross_loss_usd else 999.0
        total_usd  = sum(o.get("pnl_usd", 0) for o in out)

        streak = 0
        if out:
            sign = 1 if _won(out[-1]) else -1
            for o in reversed(out):
                if _won(o) == (sign == 1):
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
            "total_pnl_usd":      round(total_usd, 2),
            "start_balance":      round(PAPER_START_BALANCE, 2),
            "paper_balance":      round(self._paper_balance, 2),
            "return_pct":         round((self._paper_balance / PAPER_START_BALANCE - 1) * 100, 2)
                                  if PAPER_START_BALANCE else 0.0,
            "streak":             streak,
            "pending":            len(self._pending),
            "total_signals":      total_fired,
            "signals_per_day":    self.signals_per_day(),
            "strategy_breakdown": self.strategy_stats(days=7),
            "recent":             out[-10:],
        }
