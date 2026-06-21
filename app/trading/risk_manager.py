"""Risk management: position sizing, stop-loss, and max drawdown guard."""
from dataclasses import dataclass, field
from typing import List, Optional
import time


def _day_key(ts: Optional[float] = None) -> str:
    """UTC calendar-day key (YYYY-MM-DD) used to anchor the daily circuit breaker."""
    return time.strftime("%Y-%m-%d", time.gmtime(ts if ts is not None else time.time()))


@dataclass
class Position:
    symbol: str
    side: str          # 'long' | 'short'
    entry_price: float
    amount: float      # CURRENT open size (shrinks after a partial close)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: int = field(default_factory=lambda: int(time.time()))
    # ── Partial-close / breakeven state (Global Risk) ──
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    partial_pct: float = 0.5
    tp1_hit: bool = False
    full_amount: float = 0.0     # original size at open (for partial math)
    contract_size: float = 1.0   # base-asset units per contract (for whole-contract partials)
    # Health monitor fields
    one_r: float = 0.0           # original 1R distance (SL size at entry)
    health_label: str = "NEUTRAL"
    health_checks: int = 0       # number of health checks done

    @property
    def pnl_pct(self) -> float:
        return 0.0  # filled by bot at runtime

    def stage_check(self, price: float) -> Optional[str]:
        """
        Staged exit trigger for the partial-close model. Returns one of:
          'stop_loss'    — full stop before TP1
          'tp1'          — TP1 reached (caller closes partial_pct, moves SL→breakeven)
          'breakeven'    — runner hit breakeven SL after TP1
          'take_profit2' — runner reached TP2
          None
        """
        long = self.side == "long"
        # Single-TP mode (legacy strategies / SJUTBotV3 / partial disabled): no TP1 set.
        if not self.tp1:
            if self.stop_loss and ((price <= self.stop_loss) if long else (price >= self.stop_loss)):
                return "stop_loss"
            if self.take_profit and ((price >= self.take_profit) if long else (price <= self.take_profit)):
                return "take_profit"
            return None
        if not self.tp1_hit:
            if self.stop_loss and ((price <= self.stop_loss) if long else (price >= self.stop_loss)):
                return "stop_loss"
            if (price >= self.tp1) if long else (price <= self.tp1):
                return "tp1"
            return None
        # after TP1: protected runner
        if self.stop_loss and ((price <= self.stop_loss) if long else (price >= self.stop_loss)):
            return "breakeven"
        if self.tp2 and ((price >= self.tp2) if long else (price <= self.tp2)):
            return "take_profit2"
        return None


class RiskManager:
    """
    Controls maximum risk per trade, total drawdown, and position limits.
    All limits are configurable; sensible defaults apply.
    """

    def __init__(self,
                 max_risk_per_trade_pct: float = 0.02,
                 stop_loss_pct: float = 0.03,
                 take_profit_pct: float = 0.06,
                 max_open_positions: int = 5,
                 max_drawdown_pct: float = 0.15,
                 fixed_trade_usdt: float = 0.0,  # >0 → fixed USDT margin per trade
                 leverage: int = 1,              # futures leverage (multiplies notional)
                 daily_loss_limit_pct: float = 0.05,  # circuit breaker: halt new entries if day PnL ≤ -5%
                 ):
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.fixed_trade_usdt = fixed_trade_usdt
        self.leverage = max(leverage, 1)
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_open_positions = max_open_positions
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self._positions: dict[str, Position] = {}
        self._peak_balance: float = 0.0
        self._halted: bool = False
        # Daily circuit breaker state (anchored to UTC calendar day)
        self._day: str = _day_key()
        self._day_start_balance: float = 0.0
        self._day_realized_pnl: float = 0.0

    def update_peak(self, balance: float):
        if balance > self._peak_balance:
            self._peak_balance = balance

    def check_drawdown(self, current_balance: float) -> bool:
        """Returns True if within drawdown limit (trading allowed)."""
        if self._peak_balance == 0:
            return True
        drawdown = (self._peak_balance - current_balance) / self._peak_balance
        if drawdown >= self.max_drawdown_pct:
            self._halted = True
            return False
        return True

    def size_position(self, balance: float, price: float) -> float:
        """Calculate position size in base asset units.

        Fixed mode  (fixed_trade_usdt > 0): margin = min(fixed, 95% of free balance).
        Percent mode (default):             margin = balance × risk_pct.
        Notional = margin × leverage.
        """
        if price <= 0:
            return 0
        if self.fixed_trade_usdt > 0:
            # Cap at 95% of available balance so we never attempt to use more than we have
            margin = min(self.fixed_trade_usdt, balance * 0.95)
        else:
            margin = balance * self.max_risk_per_trade_pct
        notional = margin * self.leverage
        return round(notional / price, 6)

    def size_by_risk(self, balance: float, price: float,
                     sl_dist_pct: float, risk_pct: float = 0.02) -> float:
        """
        Dynamic sizing: choose notional so the worst-case loss (full size hitting
        the initial SL) ≈ risk_pct of the portfolio.

          risk_amount = balance × risk_pct
          loss_at_SL  = notional × sl_dist_pct   →   notional = risk_amount / sl_dist_pct

        Capped at 95% of balance × leverage (can't exceed available margin).
        Falls back to fixed/percent sizing if sl_dist_pct is unusable.
        """
        if price <= 0:
            return 0.0
        if sl_dist_pct and sl_dist_pct > 0:
            risk_amount = balance * max(risk_pct, 0.0)
            notional    = risk_amount / sl_dist_pct
            max_notional = balance * 0.95 * self.leverage
            notional    = min(notional, max_notional)
            return round(notional / price, 6)
        # Fallback: existing fixed/percent sizing
        return self.size_position(balance, price)

    # ── Daily circuit breaker ──────────────────────────────────────────────

    def _roll_day(self, balance: float):
        """Reset daily PnL accounting when the UTC calendar day changes."""
        today = _day_key()
        if today != self._day or self._day_start_balance <= 0:
            self._day = today
            self._day_start_balance = balance if balance > 0 else self._day_start_balance
            self._day_realized_pnl = 0.0

    def register_pnl(self, pnl: float):
        """Record a realized PnL (per closed position / partial close) for the day."""
        self._day_realized_pnl += pnl

    def check_daily_circuit(self, balance: float) -> tuple[bool, str]:
        """
        Returns (allowed, reason). New entries are blocked for the rest of the
        UTC day once realized PnL ≤ -daily_loss_limit_pct of the day-start balance.
        """
        self._roll_day(balance)
        if self._day_start_balance <= 0 or self.daily_loss_limit_pct <= 0:
            return True, "ok"
        loss_pct = self._day_realized_pnl / self._day_start_balance
        if loss_pct <= -self.daily_loss_limit_pct:
            return False, (f"Daily circuit breaker: PnL {loss_pct*100:.1f}% "
                           f"≤ -{self.daily_loss_limit_pct*100:.0f}% — paused until next day")
        return True, "ok"

    def compute_stops(self, side: str, entry_price: float) -> tuple[float, float]:
        """Returns (stop_loss_price, take_profit_price). Accepts 'buy'/'long' or 'sell'/'short'."""
        if side in ("buy", "long"):
            sl = entry_price * (1 - self.stop_loss_pct)
            tp = entry_price * (1 + self.take_profit_pct)
        else:
            sl = entry_price * (1 + self.stop_loss_pct)
            tp = entry_price * (1 - self.take_profit_pct)
        return round(sl, 6), round(tp, 6)

    def can_open(self, symbol: str, strategy: str = "") -> tuple[bool, str]:
        if self._halted:
            return False, "Trading halted: max drawdown reached"
        key = f"{symbol}||{strategy}"
        if key in self._positions:
            return False, f"{strategy} already has open position for {symbol}"
        # 1 position per strategy across ALL symbols (long or short counts the same)
        base = strategy[:-6] if strategy.endswith("_short") else strategy
        for k in self._positions:
            k_slot = k.split("||")[1] if "||" in k else ""
            k_base = k_slot[:-6] if k_slot.endswith("_short") else k_slot
            if k_base == base:
                return False, f"{base} already has an open position"
        if len(self._positions) >= self.max_open_positions:
            return False, f"Max open positions ({self.max_open_positions}) reached"
        return True, "ok"

    def open_position(self, symbol: str, side: str, entry_price: float, amount: float,
                      strategy: str = "", stop_loss: float = None, take_profit: float = None,
                      tp1: float = None, tp2: float = None, partial_pct: float = 0.5,
                      contract_size: float = 1.0, one_r: float = 0.0) -> Position:
        if stop_loss is None or take_profit is None:
            sl_default, tp_default = self.compute_stops(side, entry_price)
            stop_loss   = stop_loss   if stop_loss   is not None else sl_default
            take_profit = take_profit if take_profit is not None else tp_default
        pos = Position(symbol=symbol, side=side, entry_price=entry_price, amount=amount,
                       stop_loss=stop_loss, take_profit=take_profit,
                       tp1=tp1, tp2=tp2, partial_pct=partial_pct, full_amount=amount,
                       contract_size=contract_size, one_r=one_r)
        self._positions[f"{symbol}||{strategy}"] = pos
        return pos

    def get_position_obj(self, symbol: str, strategy: str = "") -> Optional[Position]:
        return self._positions.get(f"{symbol}||{strategy}")

    def close_position(self, symbol: str, strategy: str = "") -> Optional[Position]:
        key = f"{symbol}||{strategy}"
        if key in self._positions:
            return self._positions.pop(key)
        # Fallback: close any position for symbol if no strategy given
        for k in list(self._positions):
            if k.startswith(f"{symbol}||"):
                return self._positions.pop(k)
        return None

    def check_stops(self, symbol: str, price: float, strategy: str = "") -> Optional[str]:
        """Returns 'stop_loss', 'take_profit', or None."""
        pos = self._positions.get(f"{symbol}||{strategy}")
        if not pos:
            return None
        if pos.side == "long":
            if pos.stop_loss and price <= pos.stop_loss:
                return "stop_loss"
            if pos.take_profit and price >= pos.take_profit:
                return "take_profit"
        else:
            if pos.stop_loss and price >= pos.stop_loss:
                return "stop_loss"
            if pos.take_profit and price <= pos.take_profit:
                return "take_profit"
        return None

    def get_positions(self) -> list[dict]:
        result = []
        for key, p in self._positions.items():
            strategy = key.split("||")[1] if "||" in key else ""
            result.append({
                "symbol": p.symbol,
                "strategy": strategy,
                "side": p.side,
                "entry": p.entry_price,
                "amount": p.amount,
                "stop_loss": p.stop_loss,
                "take_profit": p.take_profit,
                "tp1": p.tp1,
                "tp2": p.tp2,
                "tp1_hit": p.tp1_hit,
                "partial_pct": p.partial_pct,
                "full_amount": p.full_amount,
                "one_r": p.one_r,
                "health_label": p.health_label,
                "health_checks": p.health_checks,
            })
        return result

    @property
    def is_halted(self) -> bool:
        return self._halted
