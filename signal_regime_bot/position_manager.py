"""
Position Manager.

Design: the SL/TP math and the health-monitor scoring are PURE functions
(no exchange calls) so backtest.py can import and reuse them exactly —
live and backtest can never compute a different stop or a different health
verdict from the same inputs. `PositionManager` wraps those pure functions
with the actual order-execution side (live only).

Health monitor rewrite (per spec): built FROM the strategy's own engines
(regime + bias re-evaluation), NOT a separate ad-hoc guard stack. A single
WEAK tier — no immediate "strong weak" — requires `weak_confirm_bars`
(default 3) CONSECUTIVE closed-30m-bar weak reads before force-closing.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import indicators as ind
from config import Config
from exchange_client import ExchangeClient
from risk_manager import RiskManager
from regime_engine import RegimeResult, BULL_LABELS, BEAR_LABELS
from bias_engine import BiasResult, BIAS_BULL, BIAS_BEAR

logger = logging.getLogger("position_manager")

LONG = "long"
SHORT = "short"


# ── Position state ────────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol: str
    side: str                  # 'long' | 'short'
    entry_price: float
    amount: float               # current open size (base units)
    full_amount: float
    stop_loss: float
    tp1: Optional[float]
    tp2: float
    one_r: float                 # initial |entry - sl|
    tp1_hit: bool = False
    realized_pnl: float = 0.0    # accumulated pnl from partial closes (TP1 leg)
    opened_at: float = field(default_factory=time.time)
    regime_at_entry: str = ""
    bias_at_entry: str = ""
    entry_score: float = 0.0
    weak_count: int = 0
    last_health_bar_ts: Optional[pd.Timestamp] = None


# ── Pure SL/TP math (shared with backtest.py) ────────────────────────────────

def calc_stop_loss(direction: str, entry: float, atr_val: float, atr_mult: float,
                   swing_high: float, swing_low: float,
                   sl_min_pct: float = 0.0, sl_max_pct: float = 1.0,
                   sl_tighten_mult: float = 1.0) -> float:
    """
    ATR-based SL, tightened to the nearest swing level if that's safer (closer),
    then clamped to [sl_min_pct, sl_max_pct] of entry price, then optionally
    pulled in further by `sl_tighten_mult` (e.g. 0.85 = stop 15% closer to
    entry than the ATR/swing/floor calc alone would place it) — this is the
    unit TP1/TP2 R-multiples are measured against, so tightening it changes
    both risk-per-trade-in-price-terms AND how close the R-multiple targets
    sit, not just the stop.

    The floor matters more than it looks: on quiet 30m candles ATR*1.5 can come
    out well under 0.1% of price. With TP1=0.5R/TP2=1.2R that shrinks the dollar
    profit target below round-trip fees on the (leverage-capped, correspondingly
    huge) notional — a "technical win" (TP2 hit) can still net a fee-driven LOSS.
    Flooring sl_dist_pct guarantees every R-multiple target clears fees with margin.
    """
    if direction == LONG:
        atr_sl = entry - atr_val * atr_mult
        sl = float(swing_low) if (not np.isnan(swing_low) and swing_low > atr_sl) else float(atr_sl)
        dist = max(entry * sl_min_pct, min(entry - sl, entry * sl_max_pct)) * sl_tighten_mult
        return entry - dist
    else:
        atr_sl = entry + atr_val * atr_mult
        sl = float(swing_high) if (not np.isnan(swing_high) and swing_high < atr_sl) else float(atr_sl)
        dist = max(entry * sl_min_pct, min(sl - entry, entry * sl_max_pct)) * sl_tighten_mult
        return entry + dist


def calc_take_profits(direction: str, entry: float, sl: float,
                      tp1_r: float, tp2_r: float) -> tuple[float, float]:
    one_r = abs(entry - sl)
    if direction == LONG:
        return entry + tp1_r * one_r, entry + tp2_r * one_r
    return entry - tp1_r * one_r, entry - tp2_r * one_r


# ── Health monitor (pure) ─────────────────────────────────────────────────────

@dataclass
class HealthResult:
    score: float
    label: str          # 'HEALTHY' | 'WEAK'
    components: dict = field(default_factory=dict)


def evaluate_health(pos: Position, df_30m: pd.DataFrame, bias: BiasResult,
                    regime: RegimeResult, cfg: Config) -> HealthResult:
    is_long = pos.side == LONG
    comps = {}

    bias_ok = (bias.bias == BIAS_BULL) if is_long else (bias.bias == BIAS_BEAR)
    comps["bias_aligned"] = 40.0 if bias_ok else 0.0

    regime_ok = regime.label in (BULL_LABELS if is_long else BEAR_LABELS)
    comps["regime_ok"] = 30.0 if regime_ok else 0.0

    macd_ok = False
    ema_ok = False
    if len(df_30m) >= max(cfg.entry_macd_slow, cfg.entry_ema_ref) + 5:
        closes = df_30m["close"]
        _, _, hist = ind.macd(closes, cfg.entry_macd_fast, cfg.entry_macd_slow, cfg.entry_macd_signal)
        h_now = float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else 0.0
        h_prev = float(hist.iloc[-2]) if not np.isnan(hist.iloc[-2]) else 0.0
        macd_ok = (h_now > h_prev) if is_long else (h_now < h_prev)

        e15 = float(ind.ema(closes, cfg.entry_ema_ref).iloc[-1])
        price = float(closes.iloc[-1])
        ema_ok = (price > e15) if is_long else (price < e15)

    comps["macd_favorable"] = 15.0 if macd_ok else 0.0
    comps["ema15_favorable"] = 15.0 if ema_ok else 0.0

    score = sum(comps.values())
    label = "HEALTHY" if score >= cfg.health_score_min else "WEAK"
    return HealthResult(score=score, label=label, components=comps)


# ── Live position manager ─────────────────────────────────────────────────────

class PositionManager:
    def __init__(self, cfg: Config, client: ExchangeClient, risk: RiskManager):
        self.cfg = cfg
        self.client = client
        self.risk = risk
        self._positions: dict[str, Position] = {}   # key: symbol (1 position per symbol, either side)

    def get(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def open_position_count(self) -> int:
        return len(self._positions)

    async def verify_no_stale_position(self, symbol: str) -> bool:
        """
        Rule #7: check the REAL OKX position before opening. Returns True if
        it's safe to open (no real position exists on either leg).
        """
        for side in (LONG, SHORT):
            amt = await self.client.fetch_position_amount(symbol, side)
            if amt > 0:
                logger.warning("[POS] %s %s has a live OKX position (%.6f) not tracked internally "
                               "— refusing to open a new one until resolved.", symbol, side, amt)
                return False
        return True

    async def reconcile_with_exchange(self, symbols: list[str]) -> list[str]:
        """
        Adopt any OKX position that already exists but isn't in internal
        state — the ONLY state this bot keeps is in-memory, so a Railway
        restart (redeploy, crash, OOM) would otherwise orphan every open
        position: nobody manages its SL/TP/health going forward, it just
        sits there protected by whatever exchange-side algo was last
        attached. Call this once at startup before the main loop.

        We don't know the original TP1/entry_score/regime — an adopted
        position resumes in single-TP mode (tp1=None, tp1_hit=True) using
        whatever SL/TP OKX currently has attached as (stop_loss, tp2).
        Returns the list of symbols adopted, for the startup log/alert.
        """
        adopted = []
        for symbol in symbols:
            if self.has_position(symbol):
                continue
            for side in (LONG, SHORT):
                details = await self.client.fetch_position_details(symbol, side)
                if not details or details["amount"] <= 0:
                    continue
                sl_price, tp_price = await self.client.fetch_attached_stops(symbol, side)
                entry_price = details["entry_price"]
                amount = details["amount"]
                # Fall back to a wide ATR-independent guess only if OKX has no
                # attached stop at all (shouldn't happen — every entry attaches
                # one — but never leave a position with sl=None, which would
                # make every price "past SL").
                if sl_price is None:
                    sl_price = entry_price * (0.9 if side == LONG else 1.1)
                    logger.warning("[RECONCILE] %s %s: no attached SL found on OKX — "
                                  "using a 10%% fallback stop until health monitor takes over",
                                  symbol, side)
                if tp_price is None:
                    tp_price = entry_price * (1.1 if side == LONG else 0.9)

                pos = Position(
                    symbol=symbol, side=side, entry_price=entry_price, amount=amount,
                    full_amount=amount, stop_loss=sl_price, tp1=None, tp2=tp_price,
                    one_r=abs(entry_price - sl_price), tp1_hit=True,
                    regime_at_entry="ADOPTED", bias_at_entry="ADOPTED",
                )
                self._positions[symbol] = pos
                adopted.append(f"{symbol} {side.upper()}")
                logger.warning("[RECONCILE] Adopted orphaned %s %s position: entry=%.6f "
                              "amount=%.6f SL=%.6f TP=%.6f (from a prior run — resuming "
                              "in single-TP mode)", symbol, side, entry_price, amount,
                              sl_price, tp_price)
        return adopted

    async def open_position(self, symbol: str, direction: str, price: float,
                            df_30m: pd.DataFrame, regime: RegimeResult, bias: BiasResult,
                            entry_score: float) -> Optional[Position]:
        c = self.cfg
        if self.has_position(symbol):
            logger.debug("[POS] %s already has a tracked position — skip", symbol)
            return None
        if not await self.verify_no_stale_position(symbol):
            return None

        side = LONG if direction == "LONG" else SHORT
        atr_val = float(ind.atr(df_30m, c.sl_atr_period).iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            logger.warning("[POS] %s ATR unavailable — skip entry", symbol)
            return None
        swing_high, swing_low = ind.recent_swing_levels(
            df_30m["high"], df_30m["low"], c.swing_lookback_left, c.swing_lookback_right)

        sl = calc_stop_loss(side, price, atr_val, c.sl_atr_mult, swing_high, swing_low,
                           c.sl_min_pct, c.sl_max_pct, c.sl_tighten_mult)
        tp1, tp2 = calc_take_profits(side, price, sl, c.tp1_r, c.tp2_r)

        # Explicit balance check BEFORE sizing — always fetch fresh (never cached/stale),
        # log it so every entry is auditable in Railway logs, and refuse to size off
        # a zero/failed balance read instead of silently producing a 0-amount order.
        balance = await self.client.fetch_balance_usdt()
        if balance <= 0:
            logger.warning("[POS] %s balance check returned %.2f USDT — refusing to size, skip entry",
                          symbol, balance)
            return None
        effective_risk_pct = c.risk_per_trade * regime.size_multiplier
        amount = self.risk.size_by_risk(balance, price, sl, regime.size_multiplier)
        logger.info("[POS] %s balance=%.2f USDT  risk=%.1f%% (base %.1f%% x regime mult %.2f)  "
                   "sl_dist=%.3f%%  -> amount=%.6f",
                   symbol, balance, effective_risk_pct * 100, c.risk_per_trade * 100,
                   regime.size_multiplier, abs(price - sl) / price * 100, amount)
        if amount <= 0:
            logger.warning("[POS] %s sizing produced 0 amount — skip entry", symbol)
            return None

        okx_side = "buy" if side == LONG else "sell"
        try:
            order = await self.client.create_order(
                symbol, okx_side, amount, pos_side=side, tp_price=tp2, sl_price=sl)
        except Exception as e:
            logger.error("[POS] %s open order failed: %s", symbol, e)
            return None

        pos = Position(
            symbol=symbol, side=side, entry_price=price, amount=order.amount,
            full_amount=order.amount, stop_loss=sl, tp1=tp1, tp2=tp2,
            one_r=abs(price - sl), regime_at_entry=regime.name,
            bias_at_entry=(bias.bias if bias is not None else regime.style),
            entry_score=entry_score,
        )
        self._positions[symbol] = pos
        logger.info("[POS] OPENED %s %s @ %.6f  SL=%.6f TP1=%.6f TP2=%.6f  amount=%.6f",
                   symbol, side.upper(), price, sl, tp1, tp2, order.amount)
        return pos

    async def check_exits_live(self, symbol: str, current_price: float) -> Optional[dict]:
        """Poll-interval tick check against live ticker price. Returns an event dict or None."""
        pos = self._positions.get(symbol)
        if pos is None:
            return None
        is_long = pos.side == LONG

        sl_hit = (current_price <= pos.stop_loss) if is_long else (current_price >= pos.stop_loss)
        if sl_hit:
            return await self._close_full(pos, current_price, "SL_HIT" if not pos.tp1_hit else "BE_HIT")

        if not pos.tp1_hit and pos.tp1 is not None:
            tp1_hit = (current_price >= pos.tp1) if is_long else (current_price <= pos.tp1)
            if tp1_hit:
                return await self._close_partial_tp1(pos, current_price)

        tp2_hit = (current_price >= pos.tp2) if is_long else (current_price <= pos.tp2)
        if tp2_hit:
            return await self._close_full(pos, current_price, "TP2_HIT")

        return None

    @staticmethod
    def _is_no_position_error(e: Exception) -> bool:
        """OKX 51169: 'you don't have any positions in this direction to reduce
        or close' — the exchange-side algo (attached SL/TP, or the OCO re-armed
        by move_sl_to_breakeven) already closed the position before our own
        price check could. This is expected under normal operation, not a
        real failure — treat it as confirmation the position is already gone."""
        s = str(e)
        return "51169" in s or "don't have any positions" in s.lower()

    async def _sync_closed_externally(self, pos: Position, price: float, reason: str) -> Optional[dict]:
        """
        Called after a close order is rejected with OKX 51169. Confirms via
        fetch_position_amount (ground truth) that the position is really gone,
        then cleans up internal state and books an approximate PnL using the
        last known price — the exact algo fill price isn't available to us,
        so this is a best-effort estimate, not a source-of-truth close price.
        Returns None (caller should retry next tick) if OKX still shows the
        position open — the 51169 was then a different, transient problem.
        """
        actual = await self.client.fetch_position_amount(pos.symbol, pos.side)
        if actual > 0:
            logger.warning("[POS] %s %s: close failed with 'no position' but OKX still shows "
                           "%.6f open — treating as transient, will retry", pos.symbol, reason, actual)
            return None

        pnl_mult = 1 if pos.side == LONG else -1
        pnl = pnl_mult * (price - pos.entry_price) * pos.amount
        balance = await self.client.fetch_balance_usdt()
        self.risk.register_trade_result(pnl, balance, time.time())
        del self._positions[pos.symbol]
        logger.info("[POS] %s %s: confirmed closed externally (exchange algo) — "
                   "synced internal state, pnl≈%.2f (approximate, exact fill unknown)",
                   pos.symbol, reason, pnl)
        return {"event": reason, "symbol": pos.symbol, "side": pos.side, "price": price,
               "pnl": pnl, "trade_pnl": pos.realized_pnl + pnl, "tp1_hit": pos.tp1_hit,
               "entry_price": pos.entry_price, "position": pos, "approximate": True}

    async def _close_partial_tp1(self, pos: Position, price: float) -> dict:
        close_amt = round(pos.full_amount * self.cfg.tp1_fraction, 8)
        close_amt = min(close_amt, pos.amount)
        okx_side = "sell" if pos.side == LONG else "buy"
        try:
            await self.client.create_order(pos.symbol, okx_side, close_amt,
                                           pos_side=pos.side, reduce_only=True)
        except Exception as e:
            if self._is_no_position_error(e):
                # The exchange-side algo closed the FULL position before our TP1
                # partial could fire — there's no partial left to take, sync as a
                # full close instead of retrying a partial that can never succeed.
                synced = await self._sync_closed_externally(pos, price, "TP1_THEN_EXTERNAL_CLOSE")
                if synced:
                    return synced
                return {"event": "ERROR", "symbol": pos.symbol, "detail": f"TP1 close failed: {e}"}
            logger.warning("[POS] %s TP1 partial close failed, retry next tick: %s", pos.symbol, e)
            return {"event": "ERROR", "symbol": pos.symbol, "detail": f"TP1 close failed: {e}"}

        pnl_mult = 1 if pos.side == LONG else -1
        pnl = pnl_mult * (price - pos.entry_price) * close_amt
        self.risk.register_trade_result(pnl, await self.client.fetch_balance_usdt(), time.time())

        pos.amount = round(pos.amount - close_amt, 8)
        pos.tp1_hit = True
        pos.realized_pnl += pnl
        pos.stop_loss = pos.entry_price   # exact breakeven

        sl_ok = await self.client.move_sl_to_breakeven(
            pos.symbol, pos.side, pos.entry_price, pos.amount, tp_price=pos.tp2)

        return {
            "event": "TP1_HIT", "symbol": pos.symbol, "side": pos.side, "price": price,
            "pnl": pnl, "sl_moved": sl_ok, "new_sl": pos.stop_loss, "position": pos,
        }

    async def _close_full(self, pos: Position, price: float, reason: str) -> dict:
        okx_side = "sell" if pos.side == LONG else "buy"
        try:
            await self.client.create_order(pos.symbol, okx_side, pos.amount,
                                           pos_side=pos.side, reduce_only=True)
        except Exception as e:
            if self._is_no_position_error(e):
                synced = await self._sync_closed_externally(pos, price, reason)
                if synced:
                    return synced
            logger.warning("[POS] %s %s close failed: %s", pos.symbol, reason, e)
            return {"event": "ERROR", "symbol": pos.symbol, "detail": f"{reason} close failed: {e}"}

        pnl_mult = 1 if pos.side == LONG else -1
        pnl = pnl_mult * (price - pos.entry_price) * pos.amount
        balance = await self.client.fetch_balance_usdt()
        self.risk.register_trade_result(pnl, balance, time.time())

        del self._positions[pos.symbol]
        return {"event": reason, "symbol": pos.symbol, "side": pos.side, "price": price,
               "pnl": pnl, "trade_pnl": pos.realized_pnl + pnl, "tp1_hit": pos.tp1_hit,
               "entry_price": pos.entry_price, "position": pos}

    async def process_closed_bar_health(self, symbol: str, df_30m: pd.DataFrame,
                                        regime: RegimeResult, bias: BiasResult) -> Optional[dict]:
        """
        Call ONCE per newly-closed 30m bar per symbol. Evaluates health and
        force-closes only after `weak_confirm_bars` CONSECUTIVE weak reads.
        """
        pos = self._positions.get(symbol)
        if pos is None or len(df_30m) == 0:
            return None
        bar_ts = df_30m.index[-1]
        if pos.last_health_bar_ts is not None and bar_ts <= pos.last_health_bar_ts:
            return None   # already evaluated this closed bar
        pos.last_health_bar_ts = bar_ts

        result = evaluate_health(pos, df_30m, bias, regime, self.cfg)
        if result.label == "HEALTHY":
            pos.weak_count = 0
            return {"event": "HEALTH_OK", "symbol": symbol, "score": result.score}

        pos.weak_count += 1
        if pos.weak_count < self.cfg.weak_confirm_bars:
            return {"event": "HEALTH_WEAK", "symbol": symbol, "score": result.score,
                   "weak_count": pos.weak_count, "confirm_needed": self.cfg.weak_confirm_bars}

        price = float(df_30m["close"].iloc[-1])
        ev = await self._close_full(pos, price, "HEALTH_CLOSE")
        ev["health_score"] = result.score
        ev["weak_count"] = pos.weak_count
        return ev
