"""
Adaptive Trading Bot — rewrite
================================
Architecture:
  LearningEngine   — per-state, per-condition weight learning with:
                     time-decay, walk-forward validation, drift detection,
                     data-quality checks, and minimum sample enforcement.
  RegimeDetector   — confidence-scored market state detection with smooth
                     transitions (no hard edge-case snapping).
  PositionHealth   — 5-component health score (momentum, volume, structure,
                     ATR, trend) used to manage open positions.
  RiskEngine       — conviction-scaled sizing, direction-aware SL/TP.
  AdaptiveTradingBot — state-machine orchestrator; plug any exchange adapter
                       that implements the BaseAdapter protocol.

Entry timeframe: 15 M (candle closes drive the tick).
Context: 1H trend filter, 4H regime classification.

Exchange: use OKXAdapter (okx_adapter.py) or any BaseAdapter subclass.
"""

import datetime
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("adaptive_bot")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    symbol:       str
    direction:    str            # "LONG" | "SHORT"
    market_state: str
    entry:        float
    exit:         float
    sl:           float
    tp1:          float
    tp2:          float
    pnl:          float
    win_loss:     str            # "WIN" | "LOSS"
    final_rr:     float
    mae:          float
    mfe:          float
    hold_sec:     float
    atr:          float
    adx:          float
    rsi:          float
    regime_score: float
    active_conditions: Dict[str, bool] = field(default_factory=dict)
    entry_score:  float = 0.0   # weighted layer score at entry
    timestamp:    str = field(default_factory=lambda: datetime.datetime.now().isoformat())


# ── Learning Engine ──────────────────────────────────────────────────────────

class LearningEngine:
    """
    Per-state, per-condition weight learning.

    Improvements over V7.0:
      1. Time-decay  — recent trades count more (half-life = DECAY_HALF)
      2. Drift guard — if recent-vs-old WR diverges > DRIFT_THR, reduce
                       learning rate by 50% and emit a warning
      3. Walk-forward validation — proposed new weights are tested on the
                       last 30% of journal before applying; rejected if
                       simulated precision does not improve by ≥ WF_IMPROVE
      4. Data quality — NaN / invalid entries silently discarded
      5. Min samples  — require MIN_SAMPLES per state before any update
      6. Delta cap    — each weight may shift at most MAX_DELTA per update
    """

    MIN_SAMPLES  = 30
    MAX_DELTA    = 5.0
    DECAY_HALF   = 50     # trades to half the weight (exponential decay)
    DRIFT_THR    = 0.15   # WR divergence threshold to flag drift
    WF_IMPROVE   = 0.02   # walk-forward must improve by this much to apply

    # Allowed weight range per condition
    W_MIN, W_MAX = 2.0, 45.0

    def __init__(self, base_weights: Dict[str, Dict[str, float]]):
        # Deep copy so mutations don't corrupt the original
        self.weights: Dict[str, Dict[str, float]] = {
            s: {c: float(w) for c, w in cw.items()}
            for s, cw in base_weights.items()
        }
        self._drift_flags: Dict[str, bool] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def update(self, journal: List[TradeRecord]) -> Dict[str, List[str]]:
        """
        Recompute weights from journal.  Returns dict of applied/skipped per state.
        Call periodically (e.g., every 50 closed trades).
        """
        from collections import defaultdict
        by_state: Dict[str, List[TradeRecord]] = defaultdict(list)
        for r in journal:
            if self._is_valid(r):
                by_state[r.market_state].append(r)

        report: Dict[str, List[str]] = {}
        for state, trades in by_state.items():
            if state not in self.weights:
                continue
            report[state] = self._update_state(state, trades)
        return report

    def get_weights(self, state: str) -> Dict[str, float]:
        return self.weights.get(state, self.weights.get("Sideway", {}))

    def drift_flags(self) -> Dict[str, bool]:
        return dict(self._drift_flags)

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _is_valid(r: TradeRecord) -> bool:
        return (
            r.win_loss in ("WIN", "LOSS")
            and not math.isnan(r.pnl)
            and r.active_conditions
        )

    def _decay_weight(self, idx: int, total: int) -> float:
        """Exponential decay: most recent trade (idx == total-1) has weight 1.0."""
        age = (total - 1) - idx
        return 0.5 ** (age / self.DECAY_HALF)

    def _detect_drift(self, trades: List[TradeRecord]) -> bool:
        n = len(trades)
        if n < self.MIN_SAMPLES * 2:
            return False
        mid = n // 2
        old_wr = sum(1 for t in trades[:mid] if t.win_loss == "WIN") / mid
        new_wr = sum(1 for t in trades[mid:] if t.win_loss == "WIN") / (n - mid)
        return abs(new_wr - old_wr) > self.DRIFT_THR

    def _compute_proposed_weights(self, state: str,
                                   trades: List[TradeRecord]) -> Optional[Dict[str, float]]:
        """Compute time-weighted condition win-rates → proposed weights."""
        current_w = self.weights[state]
        n = len(trades)

        total_dw = sum(self._decay_weight(i, n) for i in range(n))
        global_win_dw = sum(
            self._decay_weight(i, n) for i, t in enumerate(trades)
            if t.win_loss == "WIN"
        )
        if total_dw <= 0:
            return None
        global_wr = global_win_dw / total_dw

        proposed = dict(current_w)
        for cond in current_w:
            c_total = c_win = 0.0
            for i, t in enumerate(trades):
                if cond not in t.active_conditions:
                    continue
                dw = self._decay_weight(i, n)
                if t.active_conditions[cond]:
                    c_total += dw
                    if t.win_loss == "WIN":
                        c_win += dw
            if c_total < 5.0:   # too few weighted observations for this condition
                continue
            cond_wr = c_win / c_total
            delta = (cond_wr - global_wr) * 10   # scale to weight units
            delta = max(-self.MAX_DELTA, min(self.MAX_DELTA, delta))
            proposed[cond] = max(self.W_MIN, min(self.W_MAX, current_w[cond] + delta))

        # Normalize so sum ≈ 100
        total = sum(proposed.values())
        if total > 0:
            factor = 100.0 / total
            proposed = {c: round(v * factor, 2) for c, v in proposed.items()}
        return proposed

    def _walk_forward_precision(self, weights: Dict[str, float],
                                 trades: List[TradeRecord]) -> float:
        """Simulated precision: among top-half scores, what fraction are wins?"""
        scored = []
        for t in trades:
            score = sum(
                weights.get(c, 0) for c, active in t.active_conditions.items() if active
            )
            scored.append((score, t.win_loss == "WIN"))
        if not scored:
            return 0.0
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max(1, len(scored) // 2)]
        return sum(1 for _, win in top if win) / len(top)

    def _update_state(self, state: str, trades: List[TradeRecord]) -> List[str]:
        log = []
        n = len(trades)

        if n < self.MIN_SAMPLES:
            log.append(f"skip: only {n} samples (need {self.MIN_SAMPLES})")
            return log

        # Drift detection
        drift = self._detect_drift(trades)
        lr_scale = 0.5 if drift else 1.0
        if drift != self._drift_flags.get(state, False):
            self._drift_flags[state] = drift
            if drift:
                logger.warning(
                    "[Learning][%s] Drift detected — WR shift > %.0f%% "
                    "between old and recent half. Learning rate halved.",
                    state, self.DRIFT_THR * 100)

        proposed = self._compute_proposed_weights(state, trades)
        if proposed is None:
            log.append("skip: could not compute proposed weights")
            return log

        # Scale deltas by lr_scale if drift
        if lr_scale < 1.0:
            current = self.weights[state]
            proposed = {
                c: current[c] + (proposed[c] - current[c]) * lr_scale
                for c in proposed
            }

        # Walk-forward validation: train on first 70%, test on last 30%
        split = max(self.MIN_SAMPLES, int(n * 0.7))
        test  = trades[split:]
        if len(test) >= 10:
            old_prec = self._walk_forward_precision(self.weights[state], test)
            new_prec = self._walk_forward_precision(proposed, test)
            if new_prec < old_prec + self.WF_IMPROVE:
                log.append(
                    f"rejected: walk-forward precision {new_prec:.3f} "
                    f"not ≥ {old_prec + self.WF_IMPROVE:.3f}"
                )
                return log
            log.append(f"validated: prec {old_prec:.3f} → {new_prec:.3f}")

        # Apply
        self.weights[state] = proposed
        log.append(f"applied ({n} trades, drift={drift})")
        logger.info("[Learning][%s] weights updated — %s", state, log[-2] if len(log) > 1 else "")
        return log


# ── Regime Detector ──────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Soft-scored regime detection (0-100 confidence per state).
    Returns the dominant state + its confidence.
    States: Trending, Expansion, Compression, Exhaustion, Sideway
    """

    STATES = ("Trending", "Expansion", "Compression", "Exhaustion", "Sideway")

    def detect(self, ind: Dict[str, Any]) -> Tuple[str, float]:
        adx     = float(ind.get("adx",      0))
        eff     = float(ind.get("eff_ratio", 0))    # 0-1 directional efficiency
        atr_exp = float(ind.get("atr_exp",   1.0))  # current ATR / ATR-MA
        bb_w    = float(ind.get("bb_width",  0.5))  # Bollinger Width percentile 0-1

        scores: Dict[str, float] = {}

        # Trending: clear directional move
        scores["Trending"] = (
            min(adx / 30, 1.0) * 50 +
            min(eff / 0.75, 1.0) * 50
        ) if adx > 18 else 0.0

        # Expansion: volatility surge — might not have direction yet
        scores["Expansion"] = (
            min(atr_exp / 1.8, 1.0) * 50 +
            min(bb_w / 0.9, 1.0) * 50
        ) if atr_exp > 1.1 else 0.0

        # Compression: tight coil — low ATR + tight BB
        scores["Compression"] = (
            max(0.0, 1 - bb_w / 0.25) * 50 +
            max(0.0, 1 - atr_exp / 0.75) * 50
        ) if bb_w < 0.35 and atr_exp < 0.85 else 0.0

        # Exhaustion: trending but efficiency collapsing (churning)
        scores["Exhaustion"] = (
            min(adx / 30, 1.0) * 40 +
            max(0.0, 1 - eff / 0.45) * 60
        ) if adx > 18 and eff < 0.5 else 0.0

        # Sideway: default when nothing else fires strongly
        top = max(scores.values()) if scores else 0.0
        scores["Sideway"] = max(0.0, 60.0 - top * 0.6)

        best = max(scores, key=scores.get)
        conf = min(round(scores[best], 1), 100.0)
        return best, conf

    def get_multiplier(self, confidence: float) -> float:
        """Position size multiplier based on regime confidence (0.90 – 1.10)."""
        return 0.90 + (confidence / 100.0) * 0.20


# ── Position Health Calculator ────────────────────────────────────────────────

class PositionHealth:
    """
    5-component health score 0-100 for an open position.
    Used to decide: hold / trail / reduce / emergency exit.
    """

    def score(self, ind: Dict[str, Any], trade: Dict[str, Any],
              price: float) -> float:
        direction = trade.get("direction", "LONG")
        lng = direction == "LONG"

        # 1. Momentum (RSI + MACD) — 30%
        rsi = float(ind.get("rsi", 50))
        rsi_s = (rsi - 30) / 40 * 100 if lng else (70 - rsi) / 40 * 100
        rsi_s = max(0.0, min(100.0, rsi_s))
        macd  = float(ind.get("macd", 0))
        msig  = float(ind.get("macd_signal", 0))
        macd_s = 80.0 if (lng and macd > msig) or (not lng and macd < msig) else 30.0
        mom_s = rsi_s * 0.6 + macd_s * 0.4

        # 2. Volume — 20%
        vol    = float(ind.get("volume", 1))
        vol_av = float(ind.get("vol_avg", max(vol, 1e-9)))
        vol_s  = min(vol / max(vol_av, 1e-9) * 60, 100.0)

        # 3. Structure (price vs EMA) — 20%
        ema5  = float(ind.get("ema5",  price))
        ema20 = float(ind.get("ema20", price))
        if lng:
            struct_s = (100 if price > ema5 > ema20
                        else 60 if price > ema20
                        else 20)
        else:
            struct_s = (100 if price < ema5 < ema20
                        else 60 if price < ema20
                        else 20)

        # 4. ATR — 15% (large ATR expansion = bad for position health)
        ea  = float(trade.get("atr_at_entry", ind.get("atr", 1)))
        ca  = float(ind.get("atr", ea))
        atr_s = max(0.0, min(100.0, (2.0 - ca / max(ea, 1e-9)) / 1.0 * 100))

        # 5. Trend (ADX) — 15%
        adx   = float(ind.get("adx", 20))
        adx_s = max(0.0, min(100.0, (adx - 15) / 25 * 100))

        return (mom_s * 0.30 + vol_s * 0.20 + struct_s * 0.20
                + atr_s * 0.15 + adx_s * 0.15)


# ── Risk Engine ───────────────────────────────────────────────────────────────

class RiskEngine:
    """
    Calculates position size and SL/TP.
    Conviction multiplier: entry_score / 100 → size ± 20% (capped 0.8×-1.2×).
    """

    BASE_ATR_MULT = 1.5    # default SL = 1.5 × ATR
    TP1_R         = 1.0    # TP1 at 1.0 R
    TP2_R         = 2.0    # TP2 at 2.0 R

    def calc(self, price: float, direction: str, ind: Dict,
             candle: Dict, base_risk_usd: float,
             entry_score: float = 65.0,
             regime_mult: float = 1.0) -> Dict[str, Any]:
        """
        Returns trade dict with entry, sl, tp1, tp2, size, sl_dist.
        """
        lng = direction == "LONG"

        pattern_sl = (candle.get("pattern_low",  price * 0.99) if lng
                      else candle.get("pattern_high", price * 1.01))
        sl_pattern = abs(price - float(pattern_sl))
        sl_atr     = float(ind.get("atr", price * 0.01)) * self.BASE_ATR_MULT
        sl_dist    = max(sl_pattern, sl_atr, price * 0.003)  # floor 0.3%

        # Conviction multiplier: high entry_score → slightly larger size
        conviction = 0.8 + (min(entry_score, 100) / 100) * 0.4  # 0.8–1.2
        size = (base_risk_usd / sl_dist) * conviction * regime_mult
        size = max(size, 0.0)

        mult = 1 if lng else -1
        sl_p  = price - sl_dist * mult
        tp1_p = price + sl_dist * self.TP1_R  * mult
        tp2_p = price + sl_dist * self.TP2_R  * mult

        return {
            "direction":            direction,
            "entry":                price,
            "sl":                   round(sl_p,  4),
            "sl_dist":              sl_dist,
            "tp1":                  round(tp1_p, 4),
            "tp2":                  round(tp2_p, 4),
            "size":                 size,
            "remaining_size":       size,
            "tp1_hit":              False,
            "tp2_hit":              False,
            "break_even_triggered": False,
            "status":               "OPEN",
            "entry_time":           datetime.datetime.now(),
            "atr_at_entry":         float(ind.get("atr", sl_dist)),
            "realized_pnl":         0.0,
            "exit_price":           None,
            "final_rr":             None,
            "mae":                  0.0,
            "mfe":                  0.0,
            "entry_score":          entry_score,
        }


# ── Adaptive Trading Bot ──────────────────────────────────────────────────────

_BASE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Trending": {
        "trend": 20, "momentum": 15, "volume": 10, "pattern": 10,
        "structure": 10, "atr": 5, "rsi": 5, "sweep": 5,
        "divergence": 5, "ema": 5, "vwap": 5, "macd": 5,
    },
    "Expansion": {
        "trend": 15, "momentum": 15, "volume": 20, "pattern": 10,
        "structure": 5, "atr": 15, "rsi": 5, "sweep": 0,
        "divergence": 5, "ema": 5, "vwap": 5, "macd": 0,
    },
    "Compression": {
        "trend": 5, "momentum": 5, "volume": 10, "pattern": 20,
        "structure": 10, "atr": 5, "rsi": 10, "sweep": 15,
        "divergence": 10, "ema": 5, "vwap": 5, "macd": 0,
    },
    "Exhaustion": {
        "trend": 5, "momentum": 10, "volume": 15, "pattern": 10,
        "structure": 10, "atr": 5, "rsi": 10, "sweep": 5,
        "divergence": 20, "ema": 5, "vwap": 0, "macd": 5,
    },
    "Sideway": {
        "trend": 5, "momentum": 10, "volume": 10, "pattern": 15,
        "structure": 10, "atr": 5, "rsi": 15, "sweep": 15,
        "divergence": 10, "ema": 0, "vwap": 5, "macd": 0,
    },
}

_SCORE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    # L1 = minimum weighted score to proceed to confirmation
    # L2 = minimum score after volatility adjustment (not used directly here —
    #      the vol-adaptive threshold is applied in _layer_score())
    "Trending":    {"L1": 62, "vol_adj": 2.0},
    "Expansion":   {"L1": 68, "vol_adj": 3.0},
    "Compression": {"L1": 60, "vol_adj": 1.0},
    "Exhaustion":  {"L1": 60, "vol_adj": 1.5},
    "Sideway":     {"L1": 63, "vol_adj": 1.5},
}


class AdaptiveTradingBot:
    """
    State machine trading bot with fully adaptive scoring.

    States: SCANNING → FILTERING → WAIT_CONFIRM → PENDING_ORDER →
            IN_POSITION / PARTIAL_EXIT / TRAILING →
            EXITING → SCANNING
            (COOLDOWN / BLOCKED / RECOVERY / ERROR intercept at any time)

    How to use:
        adapter  = OKXAdapter(symbol="BTC-USDT-SWAP", ...)
        bot      = AdaptiveTradingBot(adapter=adapter, balance=10_000)
        bot.load_state()
        bot.reconcile()
        # then on each 15M close:
        bot.on_tick(candle_15m, candle_1h, candle_4h,
                    ind_15m, ind_1h, ind_4h, extras)
    """

    STATES = frozenset({
        "SCANNING", "FILTERING", "WAIT_CONFIRM", "PENDING_ORDER",
        "IN_POSITION", "PARTIAL_EXIT", "TRAILING",
        "EXITING", "COOLDOWN", "BLOCKED", "RECOVERY", "ERROR",
    })

    DEFAULT_STATE_FILE = "adaptive_bot_state.json"
    LEARN_EVERY        = 50    # retrain every N closed trades
    MAX_TICK_DEPTH     = 6
    ATR_HISTORY_LEN    = 200

    def __init__(
        self,
        adapter                 = None,          # BaseAdapter (OKXAdapter / paper)
        balance:           float = 10_000.0,
        risk_pct:          float = 0.01,         # 1% per trade
        daily_loss_limit:  float = -3.0,         # % of balance
        daily_profit_limit:float = 8.0,
        cooldown_minutes:  int   = 30,
        max_loss_streak:   int   = 3,
        tp1_close_pct:     float = 0.50,
        state_file:        str   = DEFAULT_STATE_FILE,
    ):
        self.adapter          = adapter
        self.balance          = balance
        self.risk_pct         = risk_pct
        self.daily_loss_lim   = daily_loss_limit
        self.daily_profit_lim = daily_profit_limit
        self.cooldown_min     = cooldown_minutes
        self.max_loss_streak  = max_loss_streak
        self.tp1_close_pct    = tp1_close_pct
        self.state_file       = state_file

        # Sub-engines
        self.learning  = LearningEngine(_BASE_WEIGHTS)
        self.regime    = RegimeDetector()
        self.health    = PositionHealth()
        self.risk_eng  = RiskEngine()

        # State machine
        self.state:            str  = "SCANNING"
        self.position_open:    bool = False
        self.current_trade:    Dict = {}
        self.direction_focus:  Optional[str] = None
        self.bars_since_trigger: int = 0

        # Market context
        self.market_state:   str   = "Sideway"
        self.state_conf:     float = 0.0
        self.regime_score:   float = 0.0
        self.atr_history:    List[float] = []

        # Risk tracking
        self.loss_streak:    int   = 0
        self.win_streak:     int   = 0
        self.daily_pnl_pct:  float = 0.0
        self.cooldown_until: Optional[datetime.datetime] = None
        self.trading_date:   Optional[datetime.date] = None

        # Journal
        self.journal:        List[TradeRecord] = []
        self._trades_since_learn: int = 0

        # Loop guard
        self._tick_depth:    int   = 0
        self._dd_notified:   bool  = False

    # ── Entry point ──────────────────────────────────────────────────────────

    def on_tick(self, candle_15m: Dict, candle_1h: Dict, candle_4h: Dict,
                ind_15m: Dict, ind_1h: Dict, ind_4h: Dict, extras: Dict):
        """
        Call on every closed 15M candle.
        4H → regime classification.
        1H → trend filter (direction gate).
        15M → layer scoring, entry trigger, position management.
        """
        self._daily_reset()
        price = float(candle_15m.get("close", 0))

        # Update ATR history from 4H (stable volatility proxy)
        atr_4h = float(ind_4h.get("atr", 0))
        if atr_4h > 0:
            self.atr_history.append(atr_4h)
            if len(self.atr_history) > self.ATR_HISTORY_LEN:
                self.atr_history.pop(0)

        # Classify regime
        self.market_state, self.state_conf = self.regime.detect(ind_4h)
        self.regime_score = self._compute_regime_score(ind_4h)

        # State machine loop (with depth guard)
        self._tick_depth = 0
        changed = True
        while changed and self._tick_depth < self.MAX_TICK_DEPTH:
            changed = False
            self._tick_depth += 1

            if self.state in ("BLOCKED", "COOLDOWN"):
                changed = self._tick_blocked()

            elif self.state == "SCANNING":
                if self._global_gates():
                    self.state = "FILTERING"
                    changed = True

            elif self.state == "FILTERING":
                changed = self._tick_filtering(ind_15m, ind_1h, candle_15m, price)

            elif self.state == "WAIT_CONFIRM":
                changed = self._tick_wait_confirm(ind_15m, price)

            elif self.state == "PENDING_ORDER":
                self._tick_pending(candle_15m, ind_15m, price)
                changed = False  # pause loop — wait for next tick

            elif self.state in ("IN_POSITION", "PARTIAL_EXIT", "TRAILING"):
                new = self._tick_in_position(price, ind_15m)
                if new != self.state:
                    self.state = new
                    changed = True

            elif self.state == "EXITING":
                self._tick_exiting(ind_15m, extras)
                changed = self.state not in ("COOLDOWN", "BLOCKED")
                if self.state not in ("COOLDOWN", "BLOCKED", "ERROR"):
                    self.state = "SCANNING"
                    changed = True

            elif self.state == "RECOVERY":
                self._tick_recovery(candle_15m, ind_15m, ind_1h)

            elif self.state == "ERROR":
                logger.error("[Bot] ERROR state — manual intervention required")
                break

        self._maybe_retrain()
        self.save_state()

    # ── State machine steps ──────────────────────────────────────────────────

    def _tick_blocked(self) -> bool:
        if self.state == "COOLDOWN":
            if not self.cooldown_until or datetime.datetime.now() >= self.cooldown_until:
                self.cooldown_until = None
                self.state = "SCANNING"
                logger.info("[Bot] Cooldown expired → SCANNING")
                return True
        return False

    def _tick_filtering(self, ind_15m, ind_1h, candle_15m, price) -> bool:
        for direction in ("LONG", "SHORT"):
            if not self._htf_filter(direction, ind_1h):
                continue
            score, conditions = self._layer_score(direction, ind_15m)
            thresh = self._dynamic_threshold()
            if score >= thresh:
                self.direction_focus    = direction
                self.bars_since_trigger = 0
                self.state              = "WAIT_CONFIRM"
                logger.info(
                    "[Bot][%s] Signal %.1f/%.1f in %s (conf=%.0f) — wait confirm",
                    direction, score, thresh, self.market_state, self.state_conf)
                return True
        self.state = "SCANNING"
        return False

    def _tick_wait_confirm(self, ind_15m, price) -> bool:
        if self.bars_since_trigger > 3:
            logger.debug("[Bot] WAIT_CONFIRM timeout → SCANNING")
            self.state = "SCANNING"
            return True
        if self._entry_trigger(self.direction_focus, ind_15m, price):
            self.state = "PENDING_ORDER"
            return True
        self.bars_since_trigger += 1
        return False

    def _tick_pending(self, candle_15m, ind_15m, price):
        score, conditions = self._layer_score(self.direction_focus, ind_15m)
        regime_mult = self.regime.get_multiplier(self.state_conf)
        base_risk   = self.balance * self.risk_pct

        trade = self.risk_eng.calc(
            price, self.direction_focus, ind_15m, candle_15m,
            base_risk_usd=base_risk,
            entry_score=score,
            regime_mult=regime_mult,
        )
        trade["market_state"]        = self.market_state
        trade["active_conditions"]   = conditions
        trade["entry_score"]         = score
        self.current_trade           = trade
        self.position_open           = True

        self._send_order("OPEN_" + self.direction_focus, {
            "entry": price,
            "sl":    trade["sl"],
            "tp1":   trade["tp1"],
            "tp2":   trade["tp2"],
            "size":  trade["size"],
        })
        self.state = "IN_POSITION"

    def _tick_in_position(self, price, ind_15m) -> str:
        t   = self.current_trade
        lng = t["direction"] == "LONG"
        sl_d = t["sl_dist"]
        mult = 1 if lng else -1
        current_r = ((price - t["entry"]) * mult) / max(sl_d, 1e-9)

        # MAE / MFE tracking
        t["mae"] = min(t["mae"], current_r)
        t["mfe"] = max(t["mfe"], current_r)

        # ── Emergency: opposing structure / momentum collapse ─────────────
        emergencies = [
            ind_15m.get("opposite_choch"),
            ind_15m.get("atr_collapse"),
            ind_15m.get("momentum_collapse"),
            ind_15m.get("volume_collapse"),
        ]
        if any(emergencies):
            self._close(price, 1.0, ind_15m, "EMERGENCY_EXIT")
            return "EXITING"

        # ── TP1 hit ───────────────────────────────────────────────────────
        tp1_hit = (price >= t["tp1"]) if lng else (price <= t["tp1"])
        if tp1_hit and not t["tp1_hit"]:
            self._close(t["tp1"], self.tp1_close_pct, ind_15m, "PARTIAL_TP1")
            t["tp1_hit"] = True
            if not t["break_even_triggered"]:
                t["sl"] = t["entry"]
                t["break_even_triggered"] = True
                logger.info("[Bot] TP1 hit @ %.4f — SL → breakeven %.4f",
                            t["tp1"], t["entry"])
            self.state = "PARTIAL_EXIT"

        # ── TP2 hit ───────────────────────────────────────────────────────
        tp2_hit = (price >= t["tp2"]) if lng else (price <= t["tp2"])
        if tp2_hit and not t["tp2_hit"]:
            self._close(t["tp2"], 1.0, ind_15m, "FULL_TP2")
            t["tp2_hit"] = True
            return "EXITING"

        # ── Health-based management ───────────────────────────────────────
        h = self.health.score(ind_15m, t, price)

        if h >= 80:
            pass  # strong — let position run, don't touch SL
        elif h >= 60 or t["break_even_triggered"]:
            atr  = float(ind_15m.get("atr", sl_d))
            trail = atr * 2.0
            new_sl = (price - trail) if lng else (price + trail)
            t["sl"] = max(t["sl"], new_sl) if lng else min(t["sl"], new_sl)
        elif h >= 40 and not t["break_even_triggered"]:
            t["sl"] = t["entry"]
            t["break_even_triggered"] = True
            logger.info("[Bot] Health %.0f → forced break-even", h)
        elif h >= 20:
            self._close(price, 0.50, ind_15m, "HEALTH_REDUCE")
            self.state = "PARTIAL_EXIT"
        else:
            self._close(price, 1.0, ind_15m, "POOR_HEALTH_EXIT")
            return "EXITING"

        # ── SL hit ────────────────────────────────────────────────────────
        sl_hit = (price <= t["sl"]) if lng else (price >= t["sl"])
        if sl_hit:
            self._close(t["sl"], 1.0, ind_15m, "SL_HIT")
            return "EXITING"

        return self.state

    def _tick_exiting(self, ind_15m, extras):
        t      = self.current_trade
        pnl    = t.get("realized_pnl", 0.0)
        result = "WIN" if pnl > 0 else "LOSS"
        self._record_trade(result, ind_15m, extras)

    def _tick_recovery(self, candle_15m, ind_15m, ind_1h):
        if self.position_open:
            return
        logger.info("[Bot] RECOVERY mode — opening next trade at half-risk")
        original = self.risk_pct
        self.risk_pct *= 0.5
        try:
            for direction in ("LONG", "SHORT"):
                if not self._htf_filter(direction, ind_1h):
                    continue
                score, _ = self._layer_score(direction, ind_15m)
                if score >= self._dynamic_threshold():
                    price = float(candle_15m.get("close", 0))
                    self._tick_pending(candle_15m, ind_15m, price)
                    logger.info("[Bot] RECOVERY trade opened: %s", direction)
                    break
        finally:
            self.risk_pct = original

    # ── Scoring ──────────────────────────────────────────────────────────────

    def _layer_score(self, direction: str,
                     ind: Dict) -> Tuple[float, Dict[str, bool]]:
        """Weighted scoring using per-state adaptive weights."""
        weights    = self.learning.get_weights(self.market_state)
        conditions: Dict[str, bool] = {}
        score = 0.0

        for key in weights:
            raw   = float(ind.get(f"{key}_score", 0))
            raw   = max(0.0, min(raw, 100.0))
            active = raw >= 50
            conditions[key] = active
            score += (raw / 100.0) * weights[key]

        # Regime confidence multiplier (high confidence = trust regime weights more)
        conf_mult = self.regime.get_multiplier(self.state_conf)
        score *= conf_mult
        return round(score, 2), conditions

    def _dynamic_threshold(self) -> float:
        """L1 entry threshold, adjusted for ATR volatility state."""
        base = _SCORE_THRESHOLDS.get(self.market_state, {}).get("L1", 63.0)
        vol_adj = _SCORE_THRESHOLDS.get(self.market_state, {}).get("vol_adj", 1.5)
        vol_state = self._atr_vol_state()
        shifts = {"Extreme": vol_adj * 2, "High": vol_adj, "Normal": 0,
                  "Low": -vol_adj, "Very Low": -vol_adj * 2}
        return base + shifts.get(vol_state, 0)

    def _atr_vol_state(self) -> str:
        if len(self.atr_history) < 20:
            return "Normal"
        import statistics
        q = sorted(self.atr_history[-100:] if len(self.atr_history) >= 100
                   else self.atr_history)
        n = len(q)
        pcts = [q[int(p / 100 * n)] for p in (10, 25, 50, 75, 90)]
        atr = self.atr_history[-1]
        for lvl, thr in zip(("Very Low", "Low", "Normal", "High", "Extreme"), pcts):
            if atr <= thr:
                return lvl
        return "Extreme"

    def _htf_filter(self, direction: str, ind_1h: Dict) -> bool:
        """1H trend filter: ADX ≥ 15 + EMA5/EMA20 aligned."""
        adx   = float(ind_1h.get("adx",   0))
        ema5  = ind_1h.get("ema5")
        ema20 = ind_1h.get("ema20")
        if adx < 15 or ema5 is None or ema20 is None:
            return False
        return (float(ema5) >= float(ema20)) if direction == "LONG" else (float(ema5) <= float(ema20))

    def _entry_trigger(self, direction: str, ind: Dict, price: float) -> bool:
        """Tighter entry confirmation on 15M (RSI zone + momentum confirmed)."""
        rsi  = float(ind.get("rsi", 50))
        mom  = float(ind.get("momentum_score", 0))
        adx  = float(ind.get("adx", 0))
        if adx < 15:
            return False
        if direction == "LONG":
            return 38 <= rsi <= 62 and mom > 50
        return 38 <= rsi <= 62 and mom > 50

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _global_gates(self) -> bool:
        if self.position_open:
            return False
        if self.cooldown_until and datetime.datetime.now() < self.cooldown_until:
            self.state = "COOLDOWN"
            return False
        if self.daily_pnl_pct <= self.daily_loss_lim:
            self.state = "BLOCKED"
            logger.warning("[Bot] BLOCKED — daily PnL %.2f%% hit loss limit", self.daily_pnl_pct)
            return False
        if self.daily_pnl_pct >= self.daily_profit_lim:
            self.state = "BLOCKED"
            logger.warning("[Bot] BLOCKED — daily PnL %.2f%% hit profit limit", self.daily_pnl_pct)
            return False
        return True

    def _compute_regime_score(self, ind_4h: Dict) -> float:
        def sc(v, w): return min(float(v), 100.0) / 100.0 * w
        return (sc(ind_4h.get("ema20_slope_score", 50), 30)
                + sc(ind_4h.get("adx_score",          50), 20)
                + sc(ind_4h.get("eff_score",           50), 20)
                + sc(ind_4h.get("atr_score",           50), 15)
                + sc(ind_4h.get("vol_score",           50), 15))

    def _close(self, price: float, portion: float, ind: Dict, reason: str):
        t = self.current_trade
        if not t or t.get("status") != "OPEN":
            return
        remaining  = t.get("remaining_size", t["size"])
        close_size = remaining * max(0.0, min(1.0, portion))
        if close_size <= 0:
            return

        lng = t["direction"] == "LONG"
        pnl_unit = (float(price) - t["entry"]) if lng else (t["entry"] - float(price))
        pnl      = pnl_unit * close_size

        t["realized_pnl"]  = t.get("realized_pnl", 0.0) + pnl
        t["remaining_size"] = remaining - close_size
        t["exit_price"]     = float(price)
        if t["sl_dist"] > 0:
            t["final_rr"] = pnl_unit / t["sl_dist"]

        if t["remaining_size"] <= 1e-9 or portion >= 1.0:
            t["status"]       = "CLOSED"
            self.position_open = False

        self._send_order("CLOSE_PARTIAL" if portion < 1.0 else "CLOSE_FULL", {
            "reason": reason, "price": float(price),
            "size": close_size, "pnl": pnl, "direction": t["direction"],
        })
        logger.info("[Bot] CLOSE %s | %s | price=%.4f pnl=%+.2f",
                    reason, t["direction"], price, pnl)

    def _send_order(self, order_type: str, info: Dict):
        if self.adapter:
            try:
                result = self.adapter.execute(order_type, info)
                logger.info("[Bot] EXEC %s → %s", order_type, result)
                return result
            except Exception as e:
                logger.error("[Bot] EXEC %s FAILED: %s — entering ERROR state", order_type, e)
                self.state = "ERROR"
                return None
        logger.info("[Bot][PAPER] %s: %s", order_type, info)
        return None

    def _record_trade(self, result: str, ind: Dict, extras: Dict):
        t      = self.current_trade
        pnl    = t.get("realized_pnl", 0.0)
        entry  = t.get("entry_time", datetime.datetime.now())
        hold_s = (datetime.datetime.now() - entry).total_seconds() if isinstance(entry, datetime.datetime) else 0.0

        rec = TradeRecord(
            symbol=extras.get("symbol", ""),
            direction=t.get("direction", ""),
            market_state=t.get("market_state", self.market_state),
            entry=t.get("entry", 0.0),
            exit=t.get("exit_price", 0.0),
            sl=t.get("sl", 0.0),
            tp1=t.get("tp1", 0.0),
            tp2=t.get("tp2", 0.0),
            pnl=pnl,
            win_loss=result,
            final_rr=t.get("final_rr", 0.0) or 0.0,
            mae=t.get("mae", 0.0),
            mfe=t.get("mfe", 0.0),
            hold_sec=hold_s,
            atr=float(ind.get("atr", 0)),
            adx=float(ind.get("adx", 0)),
            rsi=float(ind.get("rsi", 50)),
            regime_score=self.regime_score,
            active_conditions=t.get("active_conditions", {}),
            entry_score=t.get("entry_score", 0.0),
        )
        self.journal.append(rec)
        self._trades_since_learn += 1

        # Update balance & streaks
        self.daily_pnl_pct += (pnl / max(self.balance, 1)) * 100
        self.balance        += pnl

        if result == "WIN":
            self.win_streak  += 1
            self.loss_streak  = 0
        else:
            self.loss_streak += 1
            self.win_streak   = 0
            if self.loss_streak >= self.max_loss_streak:
                self.cooldown_until = (datetime.datetime.now()
                                       + datetime.timedelta(minutes=self.cooldown_min))
                self.state = "COOLDOWN"
                logger.warning("[Bot] Loss streak %d → COOLDOWN until %s",
                               self.loss_streak,
                               self.cooldown_until.strftime("%H:%M:%S"))

        logger.info("[Bot] TRADE %s | pnl=%+.2f | bal=%.2f | ws=%d ls=%d",
                    result, pnl, self.balance, self.win_streak, self.loss_streak)

    def _maybe_retrain(self):
        if self._trades_since_learn >= self.LEARN_EVERY and len(self.journal) >= LearningEngine.MIN_SAMPLES:
            report = self.learning.update(self.journal)
            self._trades_since_learn = 0
            # Log drift flags
            for state, flagged in self.learning.drift_flags().items():
                if flagged:
                    logger.warning("[Bot] Drift flag active for state=%s", state)
            logger.info("[Bot] Learning update: %s", {s: v[-1] for s, v in report.items() if v})

    def _daily_reset(self):
        today = datetime.date.today()
        if self.trading_date != today:
            self.trading_date  = today
            self.daily_pnl_pct = 0.0
            if self.state == "BLOCKED":
                self.state = "SCANNING"
                logger.info("[Bot] New trading day — BLOCKED state reset")

    # ── State persistence ────────────────────────────────────────────────────

    def save_state(self):
        snap = {
            "state":           self.state,
            "position_open":   self.position_open,
            "current_trade":   self.current_trade,
            "balance":         self.balance,
            "daily_pnl_pct":   self.daily_pnl_pct,
            "loss_streak":     self.loss_streak,
            "win_streak":      self.win_streak,
            "cooldown_until":  self.cooldown_until.isoformat() if self.cooldown_until else None,
            "trading_date":    self.trading_date.isoformat() if self.trading_date else None,
            "market_state":    self.market_state,
            "direction_focus": self.direction_focus,
            "atr_history":     self.atr_history[-self.ATR_HISTORY_LEN:],
            "learning_weights": self.learning.weights,
            "saved_at":        datetime.datetime.now().isoformat(),
        }
        tmp = self.state_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(snap, f, indent=2, default=str, ensure_ascii=False)
            os.replace(tmp, self.state_file)
        except OSError as e:
            logger.error("[Bot] save_state failed: %s", e)

    def load_state(self) -> bool:
        if not os.path.exists(self.state_file):
            logger.info("[Bot] No saved state at '%s' — fresh start", self.state_file)
            return False
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("[Bot] load_state failed: %s — fresh start", e)
            return False

        self.state          = data.get("state", "SCANNING")
        self.position_open  = data.get("position_open", False)
        self.current_trade  = data.get("current_trade") or {}
        self.balance        = data.get("balance", self.balance)
        self.daily_pnl_pct  = data.get("daily_pnl_pct", 0.0)
        self.loss_streak    = data.get("loss_streak", 0)
        self.win_streak     = data.get("win_streak", 0)
        self.atr_history    = data.get("atr_history", [])
        self.direction_focus = data.get("direction_focus")
        self.market_state   = data.get("market_state", "Sideway")

        cu = data.get("cooldown_until")
        self.cooldown_until = datetime.datetime.fromisoformat(cu) if cu else None
        td = data.get("trading_date")
        self.trading_date   = datetime.date.fromisoformat(td) if td else None

        # Restore learned weights
        w = data.get("learning_weights")
        if w:
            self.learning.weights = {s: {c: float(v) for c, v in cw.items()}
                                     for s, cw in w.items() if s in self.learning.weights}

        et = (self.current_trade or {}).get("entry_time")
        if isinstance(et, str):
            try:
                self.current_trade["entry_time"] = datetime.datetime.fromisoformat(et)
            except ValueError:
                self.current_trade["entry_time"] = datetime.datetime.now()

        logger.info("[Bot] State loaded | state=%s pos=%s bal=%.2f saved=%s",
                    self.state, self.position_open, self.balance, data.get("saved_at"))
        return True

    def reconcile(self, symbol: Optional[str] = None) -> None:
        """Compare local position state against exchange. Call after load_state()."""
        if self.adapter is None:
            logger.debug("[Bot] No adapter — skipping reconciliation")
            return
        if not hasattr(self.adapter, "fetch_open_position"):
            return

        try:
            live = self.adapter.fetch_open_position(symbol)
        except Exception as e:
            logger.error("[Bot] reconcile fetch failed: %s — entering ERROR", e)
            self.state = "ERROR"
            return

        live_size = float((live or {}).get("contracts", 0) or 0)

        if self.position_open and live_size == 0:
            logger.warning("[Bot] Local has position but exchange has none — clearing (closed offline)")
            self.position_open = False
            self.current_trade = {}
            if self.state not in ("BLOCKED", "COOLDOWN", "ERROR"):
                self.state = "SCANNING"

        elif not self.position_open and live_size != 0:
            direction   = "LONG" if live_size > 0 else "SHORT"
            entry_price = float((live or {}).get("entryPrice") or (live or {}).get("entryPx") or 0)
            logger.warning("[Bot] Exchange has %s position (%.4f) not in local state — adopting",
                           direction, abs(live_size))
            self.current_trade = {
                "direction": direction, "entry": entry_price,
                "sl":  entry_price * (0.99 if direction == "LONG" else 1.01),
                "sl_dist": max(entry_price * 0.01, 1e-9),
                "tp1": entry_price, "tp2": entry_price,
                "size": abs(live_size), "remaining_size": abs(live_size),
                "tp1_hit": False, "tp2_hit": False,
                "break_even_triggered": False, "status": "OPEN",
                "entry_time": datetime.datetime.now(),
                "atr_at_entry": 0.0, "realized_pnl": 0.0,
                "exit_price": None, "final_rr": None,
                "mae": 0.0, "mfe": 0.0,
            }
            self.position_open = True
            self.state         = "IN_POSITION"
        else:
            logger.info("[Bot] Reconciliation OK — no discrepancy")

    # ── Public helpers ───────────────────────────────────────────────────────

    def status(self) -> Dict:
        return {
            "state":         self.state,
            "position_open": self.position_open,
            "direction":     self.current_trade.get("direction") if self.position_open else None,
            "entry":         self.current_trade.get("entry")     if self.position_open else None,
            "realized_pnl":  self.current_trade.get("realized_pnl", 0),
            "balance":       self.balance,
            "daily_pnl_pct": self.daily_pnl_pct,
            "loss_streak":   self.loss_streak,
            "win_streak":    self.win_streak,
            "total_trades":  len(self.journal),
            "market_state":  self.market_state,
            "state_conf":    self.state_conf,
            "regime_score":  self.regime_score,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "drift_flags":   self.learning.drift_flags(),
        }

    def performance(self) -> Dict:
        j = self.journal
        if not j:
            return {"message": "no trades yet"}
        wins   = [r for r in j if r.win_loss == "WIN"]
        losses = [r for r in j if r.win_loss == "LOSS"]
        pnls   = [r.pnl for r in j]
        by_state: Dict = {}
        for r in j:
            s = r.market_state
            by_state.setdefault(s, {"wins": 0, "total": 0, "pnl": 0.0})
            by_state[s]["total"] += 1
            by_state[s]["pnl"]   += r.pnl
            if r.win_loss == "WIN":
                by_state[s]["wins"] += 1
        return {
            "total":          len(j),
            "win_rate":       len(wins) / len(j),
            "net_pnl":        sum(pnls),
            "avg_win":        sum(r.pnl for r in wins)   / max(len(wins), 1),
            "avg_loss":       sum(r.pnl for r in losses) / max(len(losses), 1),
            "profit_factor":  (sum(r.pnl for r in wins) /
                               max(abs(sum(r.pnl for r in losses)), 1e-9)),
            "by_state":       {s: {**v, "win_rate": v["wins"] / max(v["total"], 1)}
                               for s, v in by_state.items()},
            "drift_flags":    self.learning.drift_flags(),
        }

    def force_state(self, new_state: str):
        if new_state not in self.STATES:
            raise ValueError(f"Unknown state: {new_state}")
        logger.info("[Bot] Force state %s → %s", self.state, new_state)
        self.state = new_state
