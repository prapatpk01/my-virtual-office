"""Risk management: position sizing, stop-loss, and max drawdown guard."""
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class Position:
    symbol: str
    side: str          # 'long' | 'short'
    entry_price: float
    amount: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: int = field(default_factory=lambda: int(time.time()))

    @property
    def pnl_pct(self) -> float:
        return 0.0  # filled by bot at runtime


class RiskManager:
    """
    Controls maximum risk per trade, total drawdown, and position limits.
    All limits are configurable; sensible defaults apply.
    """

    def __init__(self,
                 max_risk_per_trade_pct: float = 0.05,   # 5% of balance per trade
                 stop_loss_pct: float = 0.03,            # 3% stop-loss
                 take_profit_pct: float = 0.06,          # 6% take-profit (2:1 RR)
                 max_open_positions: int = 5,
                 max_drawdown_pct: float = 0.15,         # halt if 15% drawdown
                 max_consecutive_sl: int = 3,            # cooldown after N losing closes in a row (PER SYMBOL)
                 cooldown_hours: float = 3.0,            # how long a symbol's cooldown lasts
                 post_cooldown_strict_trades: int = 5,   # tighten entries for N trades after resuming
                 post_cooldown_threshold_bonus: float = 6.0,  # extra quality points required during that window
                 ):
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_open_positions = max_open_positions
        self.max_drawdown_pct = max_drawdown_pct
        self.max_consecutive_sl = max_consecutive_sl
        self.cooldown_seconds = cooldown_hours * 3600
        self.post_cooldown_strict_trades = post_cooldown_strict_trades
        self.post_cooldown_threshold_bonus = post_cooldown_threshold_bonus
        self._positions: dict[str, Position] = {}
        self._peak_balance: float = 0.0
        self._halted: bool = False
        # Per-symbol cooldown state (silver/gold losing a run shouldn't pause BTC):
        self._sl_streak: dict[str, int] = {}       # symbol -> consecutive losing closes
        self._cooldown_until: dict[str, float] = {}  # symbol -> epoch when it resumes
        self._strict_left: dict[str, int] = {}     # symbol -> stricter entries remaining after resume

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

    def record_trade_result(self, pnl: float, symbol: str = "") -> bool:
        """
        Track consecutive losing closes PER SYMBOL. A losing close (pnl < 0)
        extends that symbol's streak; any non-losing close resets it. When a
        symbol hits max_consecutive_sl, only that symbol is paused, and its
        first post_cooldown_strict_trades entries after resuming are tightened.
        Returns True the moment this result just triggered a cooldown.
        """
        streak = self._sl_streak.get(symbol, 0)
        if pnl < 0:
            streak += 1
        else:
            streak = 0
        self._sl_streak[symbol] = streak

        if streak >= self.max_consecutive_sl:
            self._cooldown_until[symbol] = time.time() + self.cooldown_seconds
            self._sl_streak[symbol] = 0
            self._strict_left[symbol] = self.post_cooldown_strict_trades
            return True
        return False

    def in_cooldown(self, symbol: str = "") -> tuple[bool, float]:
        """Returns (is_in_cooldown, seconds_remaining) for one symbol."""
        remaining = self._cooldown_until.get(symbol, 0.0) - time.time()
        if remaining > 0:
            return True, remaining
        return False, 0.0

    def entry_threshold_bonus(self, symbol: str = "") -> float:
        """Extra quality points a symbol's entry must clear while it's in the
        post-cooldown strict window (0 when not tightened)."""
        if self._strict_left.get(symbol, 0) > 0:
            return self.post_cooldown_threshold_bonus
        return 0.0

    def consume_strict_entry(self, symbol: str = "") -> None:
        """Call once a stricter post-cooldown entry actually opens — counts it
        down so the tightening lifts after post_cooldown_strict_trades trades."""
        if self._strict_left.get(symbol, 0) > 0:
            self._strict_left[symbol] -= 1

    def size_position(self, balance: float, price: float,
                      size_pct: float = None) -> float:
        """Calculate position size.
        size_pct overrides max_risk_per_trade_pct when provided (0.08–0.12 etc.)."""
        if price <= 0:
            return 0
        pct = size_pct if size_pct is not None else self.max_risk_per_trade_pct
        return round(balance * pct / price, 6)

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
        in_cd, remaining = self.in_cooldown(symbol)
        if in_cd:
            return False, (
                f"{symbol} cooldown after {self.max_consecutive_sl} consecutive losing closes "
                f"— resumes in {remaining/60:.0f} min"
            )
        key = f"{symbol}||{strategy}"
        if key in self._positions:
            return False, f"{strategy} already has open position for {symbol}"
        sym_count = sum(1 for k in self._positions if k.startswith(f"{symbol}||"))
        if sym_count >= 2:
            return False, f"Max 2 positions per symbol for {symbol}"
        if len(self._positions) >= self.max_open_positions:
            return False, f"Max open positions ({self.max_open_positions}) reached"
        return True, "ok"

    def open_position(self, symbol: str, side: str, entry_price: float, amount: float,
                      strategy: str = "", stop_loss: float = None, take_profit: float = None) -> Position:
        if stop_loss is None or take_profit is None:
            sl_default, tp_default = self.compute_stops(side, entry_price)
            stop_loss  = stop_loss  or sl_default
            take_profit = take_profit or tp_default
        pos = Position(symbol=symbol, side=side, entry_price=entry_price, amount=amount,
                       stop_loss=stop_loss, take_profit=take_profit)
        self._positions[f"{symbol}||{strategy}"] = pos
        return pos

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

    def update_stop_loss(self, symbol: str, new_sl: float, strategy: str = "") -> bool:
        """Update the stop-loss price for an open position (used by trailing stop / break-even)."""
        pos = self._positions.get(f"{symbol}||{strategy}")
        if pos:
            pos.stop_loss = new_sl
            return True
        return False

    def reduce_position(self, symbol: str, amount: float, strategy: str = "") -> None:
        """Reduce position size after a partial take-profit execution."""
        pos = self._positions.get(f"{symbol}||{strategy}")
        if pos:
            pos.amount = max(0.0, round(pos.amount - amount, 8))

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
            })
        return result

    @property
    def is_halted(self) -> bool:
        return self._halted
