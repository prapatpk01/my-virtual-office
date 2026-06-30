"""
Adaptive Trading Bot — v7.0
============================
รวม AdaptiveEngine (จาก adaptive_engine_draft.py) + SJUTBotV4 signal engine เข้าด้วยกัน
แก้ไขทุกจุดที่วิเคราะห์พบ:

[FIX 1] _calculate_health() — คำนวณจริงจาก 5 ตัวชี้วัด (momentum, volume,
         structure, ATR, divergence) แทนที่ค่าคงที่แบบเดิมที่ไม่ได้คำนวณอะไรจริง
[FIX 2] _close_position() — บันทึก PnL จริง + อัปเดต current_trade + ส่งผ่าน
         execution callback ที่ wire ได้จากภายนอก
[FIX 3] adjust_weights_from_learning() — วิเคราะห์แยก per market_state
         ไม่ใช่ winrate รวมทั้งหมด
[FIX 4] Partial TP ครอบคลุมทั้ง LONG และ SHORT
[FIX 5] on_tick() while-loop มี guard ป้องกัน infinite loop
[FIX 6] _scale_score() — cap input ที่ 100 ก่อนคำนวณ
[FIX 7] TP3 hit logic ใน _manage_open_position()
[FIX 8] Recovery state มี logic ใน on_tick()
[FIX 9] cooldown_until ใช้ datetime แทน time.time() float
[FIX 10] Dynamic weight learning ปรับ per-condition ไม่ใช่ global shift

--- v6.0: production readiness สำหรับเทรดจริงบน OKX ---
[FIX 11] Logging module จริง (console + rotating file) แทน print()
[FIX 12] _send_order() แยก fatal error ออกจาก error ที่ retry แล้วหมดสิทธิ์
         (retry/backoff จริงอยู่ใน OKXExecutionAdapter — ดู okx_exchange.py)
[FIX 13] save_state() / load_state() / reconcile_with_exchange() —
         persist + reload position state ตอนรีสตาร์ตบอท พร้อมเทียบกับ
         position จริงบน exchange (กันกรณี SL/TP ถูก trigger ระหว่างบอท offline)

--- v7.0: multi-timeframe entry (15M) + TP structure ---
[FIX 17] เปลี่ยน timeframe เข้าออเดอร์เป็น 15M ทั้งหมด (layer scoring, entry
         trigger, risk engine, position management) — 4H ใช้กำหนด market_state/
         regime เหมือนเดิม, เพิ่ม 1H เป็นตัวกรองแนวโน้มชั้นกลาง
         (_check_htf_trend_filter) ก่อนอนุญาตให้สัญญาณ 15M เข้าไม้ได้
         → on_tick() ตอนนี้ต้องรับ candle_15m/ind_15m เพิ่ม (breaking change
         จาก v6 — ดู signature ใหม่)
[FIX 16] ตัด TP3 ออก เหลือแค่ TP1 (scale-out ตาม tp1_close_pct, default 50%)
         และ TP2 (ปิดที่เหลือทั้งหมด = full exit) — พอ TP1 hit จะขยับ SL ไป
         breakeven ทันที (เดิมต้องรอ current_r >= 0.8 แยกอิสระจาก TP)

หมายเหตุ: ข้อมูล indicator ตัวอย่างใน `if __name__ == "__main__":` ด้านล่าง เป็น
ตัวอย่าง indicator สำหรับรันเดโม/เทสต์เท่านั้น ไม่ใช่ placeholder ของ logic จริง —
logic ทั้งหมดด้านบนคำนวณจาก indicator ที่ส่งเข้ามาจริงทุกจุดแล้ว

ต้องการ: numpy, pandas (สำหรับ signal engine), ccxt (สำหรับ okx_exchange.py)
"""

import numpy as np
import datetime
import json
import os
import logging
from typing import Optional, Callable, Dict, List, Any


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# Railway doesn't have persistent disk — use module-level logger only (no file handler)
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger("adaptive_trading_bot")


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 2: REGIME-SPECIFIC ADAPTIVE THRESHOLDS
# ADX, RSI, direction bias, and L1 delta per 8-state market regime
# ──────────────────────────────────────────────────────────────────────────────

REGIME_THRESHOLDS: Dict[str, Dict] = {
    # Strong directional trend — LONG only, dip-buy, lower bar to enter
    "STRONG_UPTREND": {
        "adx_1h_min": 20, "rsi_long": (40, 65), "rsi_short": (35, 58),
        "allow_long": True,  "allow_short": False, "l1_delta": -5,
    },
    # Normal trend — both directions, standard RSI dip gates
    "UPTREND": {
        "adx_1h_min": 15, "rsi_long": (35, 62), "rsi_short": (38, 65),
        "allow_long": True,  "allow_short": True,  "l1_delta": 0,
    },
    # Ranging market — mean-reversion RSI gates, higher quality bar
    "RANGE": {
        "adx_1h_min": 12, "rsi_long": (28, 48), "rsi_short": (52, 72),
        "allow_long": True,  "allow_short": True,  "l1_delta": +5,
    },
    # Volatility squeeze — wait for breakout quality, tight RSI
    "LOW_VOL": {
        "adx_1h_min": 10, "rsi_long": (45, 60), "rsi_short": (40, 55),
        "allow_long": True,  "allow_short": True,  "l1_delta": +10,
    },
    # High volatility expansion — very strict ADX + tight RSI + high bar
    "HIGH_VOL": {
        "adx_1h_min": 22, "rsi_long": (42, 58), "rsi_short": (42, 58),
        "allow_long": True,  "allow_short": True,  "l1_delta": +8,
    },
    # Topping/distribution — SHORT bias, longs blocked
    "DISTRIBUTION": {
        "adx_1h_min": 15, "rsi_long": (28, 45), "rsi_short": (52, 72),
        "allow_long": False, "allow_short": True,  "l1_delta": +3,
    },
    # Bottoming/accumulation — LONG bias, shorts blocked
    "ACCUMULATION": {
        "adx_1h_min": 12, "rsi_long": (35, 55), "rsi_short": (48, 68),
        "allow_long": True,  "allow_short": False, "l1_delta": +3,
    },
    # Extended trend losing momentum — very high quality bar required
    "EXHAUSTION": {
        "adx_1h_min": 18, "rsi_long": (30, 48), "rsi_short": (52, 70),
        "allow_long": True,  "allow_short": True,  "l1_delta": +15,
    },
}

_TRADEABLE_STATES: frozenset = frozenset({
    "STRONG_UPTREND",  # trend-follow LONG — highest WR
    "UPTREND",         # trend-follow both directions
    "DISTRIBUTION",    # SHORT-bias reversal (topping pattern)
    "ACCUMULATION",    # LONG-bias reversal (bottoming pattern)
    "HIGH_VOL",        # expansion — high bar (l1_delta +8) protects quality
    "EXHAUSTION",      # elevated ADX + collapsed efficiency — l1_delta +15 ensures
                       # only very high-confidence setups enter; 1H EMA filter guards direction
    # RANGE: blocked — catch-all Sideway equivalent, WR 33-45%
    # LOW_VOL: blocked — wait for breakout, no directional edge yet
})

# Scores that are LONG-biased: high value = bullish signal.
# For SHORT scoring these must be inverted (100 - score) so bearish conditions score high.
# trend_score is intentionally excluded: it measures directional STRENGTH not bias
# (both ema_bull and ema_bear get high trend_score, which correctly favours either direction).
_DIR_INVERT_KEYS: tuple = ("momentum", "structure", "ema", "vwap", "macd")

# Entry type label per regime (for learning database enrichment)
_REGIME_ENTRY_TYPE: Dict[str, str] = {
    "STRONG_UPTREND": "trend_follow",
    "UPTREND":        "trend_follow",
    "RANGE":          "mean_revert",
    "LOW_VOL":        "breakout",
    "HIGH_VOL":       "momentum",
    "DISTRIBUTION":   "reversal",
    "ACCUMULATION":   "reversal",
    "EXHAUSTION":     "counter_trend",
}


# ──────────────────────────────────────────────────────────────────────────────
# ADAPTIVE ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveEngine:
    """
    จัดการ dynamic weights, ATR volatility state, adaptive thresholds
    และ learning จาก trade journal
    """

    def __init__(self):
        self.regime_tiers = [
            (90, 1.10), (75, 1.05), (50, 1.00), (30, 0.95), (0, 0.90)
        ]
        self.atr_percentile_points = [10, 25, 50, 75, 90]
        self.atr_vol_labels = ["Very Low", "Low", "Normal", "High", "Extreme"]

        # MODULE 1 base weights — 8 regimes, sum = 100 per state
        self.base_weights: Dict[str, Dict[str, float]] = {
            "STRONG_UPTREND": {
                "trend": 25, "momentum": 20, "volume": 10, "pattern": 8,
                "structure": 12, "atr": 5, "rsi": 5, "sweep": 0,
                "divergence": 5, "ema": 5, "vwap": 5, "macd": 0,
            },
            "UPTREND": {
                "trend": 20, "momentum": 15, "volume": 10, "pattern": 10,
                "structure": 10, "atr": 5, "rsi": 5, "sweep": 5,
                "divergence": 5, "ema": 5, "vwap": 5, "macd": 5,
            },
            "RANGE": {
                "trend": 5, "momentum": 10, "volume": 10, "pattern": 15,
                "structure": 10, "atr": 5, "rsi": 20, "sweep": 10,
                "divergence": 10, "ema": 0, "vwap": 5, "macd": 0,
            },
            "LOW_VOL": {
                "trend": 10, "momentum": 5, "volume": 5, "pattern": 20,
                "structure": 10, "atr": 20, "rsi": 5, "sweep": 10,
                "divergence": 5, "ema": 5, "vwap": 5, "macd": 0,
            },
            "HIGH_VOL": {
                "trend": 10, "momentum": 15, "volume": 20, "pattern": 10,
                "structure": 5, "atr": 15, "rsi": 5, "sweep": 5,
                "divergence": 5, "ema": 5, "vwap": 5, "macd": 0,
            },
            "DISTRIBUTION": {
                "trend": 5, "momentum": 15, "volume": 15, "pattern": 10,
                "structure": 10, "atr": 5, "rsi": 10, "sweep": 5,
                "divergence": 15, "ema": 5, "vwap": 5, "macd": 0,
            },
            "ACCUMULATION": {
                "trend": 5, "momentum": 10, "volume": 15, "pattern": 10,
                "structure": 10, "atr": 5, "rsi": 15, "sweep": 5,
                "divergence": 15, "ema": 5, "vwap": 5, "macd": 0,
            },
            "EXHAUSTION": {
                "trend": 5, "momentum": 10, "volume": 15, "pattern": 10,
                "structure": 10, "atr": 5, "rsi": 10, "sweep": 5,
                "divergence": 20, "ema": 5, "vwap": 0, "macd": 5,
            },
        }

        # MODULE 3 learning stats — tracks win/loss R-multiples per condition per state
        def _init_stats(w: Dict) -> Dict:
            return {
                "win_count": 0, "loss_count": 0,
                "sum_win_r": 0.0, "sum_loss_r": 0.0,
                "condition_wins":   {k: 0   for k in w},
                "condition_total":  {k: 0   for k in w},
                "condition_win_r":  {k: 0.0 for k in w},
                "condition_loss_r": {k: 0.0 for k in w},
            }
        self._learning_stats: Dict[str, Dict] = {
            state: _init_stats(w) for state, w in self.base_weights.items()
        }

    # ── Regime / Volatility helpers ──────────────────────────────────────────

    def get_regime_multiplier(self, regime_score: float) -> float:
        for threshold, multiplier in self.regime_tiers:
            if regime_score >= threshold:
                return multiplier
        return 0.90

    def get_atr_volatility_state(self, current_atr: float, atr_history: List[float]) -> str:
        if len(atr_history) < 20:
            return "Normal"
        history = atr_history[-100:] if len(atr_history) >= 100 else atr_history
        percentiles = np.percentile(history, self.atr_percentile_points)
        for i in range(len(percentiles) - 1, -1, -1):
            if current_atr >= percentiles[i]:
                return self.atr_vol_labels[min(i, len(self.atr_vol_labels) - 1)]
        return "Very Low"

    def get_dynamic_weights(self, market_state: str) -> Dict[str, float]:
        return self.base_weights.get(market_state, self.base_weights["UPTREND"])

    def get_regime_thresholds(self, market_state: str) -> Dict:
        return REGIME_THRESHOLDS.get(market_state, REGIME_THRESHOLDS["UPTREND"])

    def get_layer_thresholds(self, vol_state: str, market_state: str = "UPTREND") -> Dict[str, float]:
        """ATR-adaptive + regime-adaptive entry threshold"""
        l1, l2 = 60.0, 55.0  # base lowered 65→60 for better trade frequency
        vol_adj = {
            "Extreme":  (+8, +5),
            "High":     (+4, +2),
            "Normal":   ( 0,  0),
            "Low":      (-4, -2),
            "Very Low": (-6, -4),
        }
        d1, d2 = vol_adj.get(vol_state, (0, 0))
        regime_delta = REGIME_THRESHOLDS.get(market_state, {}).get("l1_delta", 0)
        return {
            "L1_PASS": l1 + d1 + regime_delta,
            "L2_PASS": l2 + d2 + max(0, regime_delta // 2),
        }

    # ── MODULE 3: Expectancy-based Learning Engine ───────────────────────────

    def record_trade_result(self, market_state: str, win: bool,
                            active_conditions: Dict[str, bool],
                            win_r: float = 1.0, loss_r: float = 1.0):
        """บันทึกผล trade + R-multiple ต่อ condition สำหรับ expectancy learning"""
        if market_state not in self._learning_stats:
            return
        stats = self._learning_stats[market_state]
        if win:
            stats["win_count"] += 1
            stats["sum_win_r"] += abs(win_r)
        else:
            stats["loss_count"] += 1
            stats["sum_loss_r"] += abs(loss_r)

        for cond, is_active in active_conditions.items():
            if cond not in stats["condition_total"]:
                continue
            stats["condition_total"][cond] += 1
            if is_active:
                if win:
                    stats["condition_wins"][cond] += 1
                    stats["condition_win_r"][cond] += abs(win_r)
                else:
                    stats["condition_loss_r"][cond] += abs(loss_r)

    def adjust_weights_from_learning(self, journal_data: List[Dict]):
        """
        MODULE 3: Expectancy-based weight adjustment
        Expectancy = WR × avg_win_R − (1−WR) × avg_loss_R
        ปรับ weight ตาม expectancy ของแต่ละ condition เทียบกับ global expectancy
        แทนที่จะดู win rate อย่างเดียว
        """
        if not journal_data:
            return

        by_state: Dict[str, List[Dict]] = {}
        for entry in journal_data:
            state = entry.get("market_state", "UPTREND")
            by_state.setdefault(state, []).append(entry)

        for state, entries in by_state.items():
            if state not in self.base_weights or len(entries) < 10:
                continue

            wins   = [e for e in entries if e.get("win_loss") == "WIN"]
            losses = [e for e in entries if e.get("win_loss") != "WIN"]
            total  = len(entries)
            global_wr = len(wins) / total

            avg_win_r_g  = sum(e.get("win_r",  1.0) for e in wins)   / max(len(wins),   1)
            avg_loss_r_g = sum(e.get("loss_r", 1.0) for e in losses)  / max(len(losses), 1)
            global_exp   = global_wr * avg_win_r_g - (1 - global_wr) * avg_loss_r_g

            stats = self._learning_stats.get(state, {})
            cond_wins_m   = stats.get("condition_wins",   {})
            cond_total_m  = stats.get("condition_total",  {})
            cond_win_r_m  = stats.get("condition_win_r",  {})
            cond_loss_r_m = stats.get("condition_loss_r", {})

            weights = self.base_weights[state]
            for cond in weights:
                ct = cond_total_m.get(cond, 0)
                if ct < 5:
                    continue
                cw = cond_wins_m.get(cond, 0)
                cl = ct - cw
                cond_wr      = cw / ct
                avg_win_r_c  = cond_win_r_m.get(cond,  0.0) / max(cw, 1)
                avg_loss_r_c = cond_loss_r_m.get(cond, 0.0) / max(cl, 1)
                cond_exp     = cond_wr * avg_win_r_c - (1 - cond_wr) * avg_loss_r_c

                # Scale by expectancy delta — positive = condition predicts good trades
                delta = (cond_exp - global_exp) * 5
                weights[cond] = float(np.clip(weights[cond] + delta, 3, 40))

            total_w = sum(weights.values())
            if total_w > 0:
                factor = 100 / total_w
                for k in weights:
                    weights[k] = round(weights[k] * factor, 1)


# ──────────────────────────────────────────────────────────────────────────────
# POSITION HEALTH CALCULATOR
# ──────────────────────────────────────────────────────────────────────────────

class PositionHealthCalculator:
    """
    [FIX 1] คำนวณ position health score 0–100 จาก 5 ตัวชี้วัด
    แทนที่ค่าคงที่แบบเดิมที่ไม่ได้คำนวณอะไรจริง
    """

    def calculate(self, ind: Dict[str, Any], trade: Dict[str, Any],
                  current_price: float) -> float:
        """
        Returns health score 0–100
        Components:
          - Momentum health (30%): RSI + MACD alignment
          - Volume health    (20%): volume vs avg volume
          - Structure health (20%): price vs key MAs
          - ATR health       (15%): ATR expansion vs entry ATR
          - Trend health     (15%): ADX strength
        """
        scores = []

        # 1. Momentum (RSI + MACD)
        rsi = ind.get("rsi", 50)
        direction = trade.get("direction", "LONG")
        if direction == "LONG":
            rsi_score = np.clip((rsi - 30) / 40 * 100, 0, 100)
        else:
            rsi_score = np.clip((70 - rsi) / 40 * 100, 0, 100)

        macd = ind.get("macd", 0)
        macd_signal = ind.get("macd_signal", 0)
        if direction == "LONG":
            macd_score = 80.0 if macd > macd_signal else 30.0
        else:
            macd_score = 80.0 if macd < macd_signal else 30.0
        momentum_score = (rsi_score * 0.6 + macd_score * 0.4)
        scores.append(("momentum", momentum_score, 0.30))

        # 2. Volume
        volume = ind.get("volume", 0)
        vol_avg = ind.get("vol_avg", volume if volume > 0 else 1)
        vol_ratio = volume / max(vol_avg, 1e-9)
        vol_score = float(np.clip(vol_ratio * 60, 0, 100))
        scores.append(("volume", vol_score, 0.20))

        # 3. Structure (price vs EMA)
        ema5 = ind.get("ema5", current_price)
        ema20 = ind.get("ema20", current_price)
        if direction == "LONG":
            struct_score = 100.0 if (current_price > ema5 and current_price > ema20) else \
                           60.0 if current_price > ema20 else 20.0
        else:
            struct_score = 100.0 if (current_price < ema5 and current_price < ema20) else \
                           60.0 if current_price < ema20 else 20.0
        scores.append(("structure", struct_score, 0.20))

        # 4. ATR health — ถ้า ATR ขยายมากกว่า entry ATR > 1.5x แสดงว่า volatility spike
        entry_atr = trade.get("atr_at_entry", ind.get("atr", 1))
        current_atr = ind.get("atr", entry_atr)
        atr_ratio = current_atr / max(entry_atr, 1e-9)
        atr_score = float(np.clip((2.0 - atr_ratio) / 1.0 * 100, 0, 100))
        scores.append(("atr", atr_score, 0.15))

        # 5. Trend (ADX)
        adx = ind.get("adx", 20)
        adx_score = float(np.clip((adx - 15) / 25 * 100, 0, 100))
        scores.append(("trend", adx_score, 0.15))

        # Weighted average
        total = sum(score * weight for _, score, weight in scores)
        return float(np.clip(total, 0, 100))


# ──────────────────────────────────────────────────────────────────────────────
# TRADING BOT
# ──────────────────────────────────────────────────────────────────────────────

class TradingBot:
    """
    State machine trading bot พร้อมใช้งาน
    ทุก bug ที่พบได้รับการแก้ไขแล้ว
    """

    # State machine states
    STATES = {
        "SCANNING", "FILTERING", "WAIT_CONFIRM", "PENDING_ORDER",
        "IN_POSITION", "PARTIAL_EXIT", "TRAILING", "EXITING",
        "COOLDOWN", "BLOCKED", "RECOVERY", "ERROR"
    }

    def __init__(self,
                 account_balance: float = 10_000.0,
                 base_risk_pct: float = 0.01,
                 daily_loss_limit_pct: float = -3.0,
                 daily_profit_limit_pct: float = 8.0,
                 cooldown_minutes: int = 20,
                 max_loss_streak: int = 4,
                 tp1_close_pct: float = 0.50,
                 state_file: Optional[str] = None,
                 execution_callback: Optional[Callable] = None,
                 startup_warmup_minutes: int = 45):
        """
        Parameters
        ----------
        tp1_close_pct : สัดส่วนที่ปิดตอน TP1 hit (ที่เหลือเป็น runner ไปจน TP2 ซึ่งปิดทั้งหมด)
            ค่า default 0.50 = ปิดครึ่งหนึ่งที่ TP1 แล้วขยับ SL ไป breakeven ทันที
            ปรับได้ตามสไตล์ — ใส่ 0.30 ถ้าอยากปิดน้อยกว่าตอน TP1 และเก็บ runner ไว้มากกว่า
        execution_callback : callable(order_type, trade_info) → None
            ฟังก์ชัน callback สำหรับส่งคำสั่งไปยัง exchange จริง
            order_type: 'OPEN_LONG' | 'OPEN_SHORT' | 'CLOSE_PARTIAL' | 'CLOSE_FULL'
            trade_info: dict ประกอบด้วย entry, sl, tp1, tp2, size, reason
            ถ้า None จะ log เท่านั้น (paper trading mode)
        """
        # Core state
        self.state: str = "SCANNING"
        self.adaptive_engine = AdaptiveEngine()
        self.health_calc = PositionHealthCalculator()
        self.tp1_close_pct = tp1_close_pct
        self._state_file: str = state_file or self.DEFAULT_STATE_FILE

        # Account
        self.account_balance = account_balance
        self.base_risk_pct = base_risk_pct

        # Risk limits
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.daily_profit_limit_pct = daily_profit_limit_pct
        self.cooldown_minutes = cooldown_minutes
        self.max_loss_streak = max_loss_streak

        # Position tracking
        self.position_open: bool = False
        self.order_status: str = "CLOSED"
        self.current_trade: Dict[str, Any] = {}

        # Market state
        self.atr_history: List[float] = []
        self.current_market_state: str = "Sideway"
        self.regime_score: float = 0.0
        self.direction_focus: Optional[str] = None
        self.bars_since_trigger: int = 0

        # Risk tracking
        self.loss_streak: int = 0
        self.win_streak: int = 0
        self.daily_pnl_pct: float = 0.0
        # [FIX 9] ใช้ datetime แทน float timestamp
        self.cooldown_until: Optional[datetime.datetime] = None
        self.trading_date: Optional[datetime.date] = None
        # Tick-scoped "now" — set at top of on_tick() so all sub-methods share same bar time
        self._bar_now: Optional[datetime.datetime] = None

        # Journal
        self.trade_journal: List[Dict] = []

        # Execution
        self.execution_callback = execution_callback

        # [FIX 5] infinite loop guard
        self._tick_depth: int = 0
        self._max_tick_depth: int = 6

        # Bar counter for holding_bars tracking
        self._bar_count: int = 0
        self._position_entry_bar: int = 0

        # Post-restart warmup — block signals for N minutes after startup
        # so ATR history, indicators and market state can stabilize
        self.startup_warmup_minutes: int = startup_warmup_minutes
        self._startup_unblock_at: Optional[datetime.datetime] = None

        # Logging
        self._log: List[str] = []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log_event(self, msg: str, level: str = "info"):
        """[FIX 11] log ผ่าน logging module จริง (console only — no file handler on Railway)"""
        log_fn = getattr(logger, level, logger.info)
        log_fn(msg)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log.append(f"[{ts}] {msg}")

    def _send_order(self, order_type: str, trade_info: Dict):
        """
        ส่งคำสั่งผ่าน execution_callback (เช่น OKXExecutionAdapter.execute จาก okx_exchange.py)
        หรือ log เป็น paper-trading ถ้าไม่มี callback

        [FIX 12] execution_callback รับผิดชอบ retry เอง (network/rate-limit errors)
        ที่นี่จับเฉพาะ exception ที่ "หลุด" ออกมาหลัง retry หมดแล้ว หรือเป็น fatal error
        (auth ผิด, fund ไม่พอ, invalid order) แล้วหยุดบอทเข้า ERROR state ทันที
        เพื่อไม่ให้ trade ต่อไปด้วยสถานะที่ไม่รู้ว่า order จริงสำเร็จหรือไม่
        """
        if self.execution_callback:
            try:
                result = self.execution_callback(order_type, trade_info)
                self._log_event(f"[EXEC OK] {order_type} confirmed → {result}")
                return result
            except Exception as e:
                self._log_event(
                    f"[ERROR] execution_callback failed for {order_type} "
                    f"(trade_info={trade_info}): {e}",
                    level="error",
                )
                self.state = "ERROR"
                return None
        else:
            self._log_event(f"[PAPER] {order_type}: {trade_info}")
            return None

    # [FIX 6] _scale_score — cap input ที่ 100
    @staticmethod
    def _scale_score(val: float, max_weight: float) -> float:
        val_clamped = min(float(val), 100.0)
        return min((val_clamped / 100.0) * max_weight, max_weight)

    # ── MODULE 1: 8-State Market Regime Engine ───────────────────────────────

    def _step1_market_state_engine(self, ind: Dict) -> str:
        """
        8-state regime detection from 4H indicators.
        Priority order: HIGH_VOL → LOW_VOL → STRONG_UPTREND → UPTREND
                      → EXHAUSTION → DISTRIBUTION → ACCUMULATION → RANGE
        """
        adx     = ind.get("adx", 0)
        eff     = ind.get("eff_ratio", 0)
        atr_exp = ind.get("atr_exp", 1.0)
        bb_w    = ind.get("bb_width", 0.5)
        pdi     = ind.get("pdi", 20)
        mdi     = ind.get("mdi", 20)
        rsi     = ind.get("rsi", 50)
        ema5    = ind.get("ema5", 0)
        ema20   = ind.get("ema20", 0)
        slope   = ind.get("ema20_slope_score", 50)  # >55 rising, <45 falling

        # Extreme volatility expansion — must trade cautiously
        if atr_exp > 1.8 and bb_w > 0.7:
            return "HIGH_VOL"

        # Volatility squeeze — pre-breakout phase
        if bb_w < 0.2 and atr_exp < 0.8:
            return "LOW_VOL"

        # Strong uptrend: efficient trend with bulls clearly dominating
        # [FIX] Checked BEFORE EXHAUSTION — a trending market with somewhat
        # low efficiency is still an UPTREND, not EXHAUSTION. Previously
        # ADX>18 + eff<0.30 fired first and blocked valid UPTREND entries.
        if adx > 22 and eff > 0.55 and pdi > mdi + 3:
            return "STRONG_UPTREND"

        # Uptrend: bulls leading with some directional efficiency
        if adx > 16 and eff > 0.35 and pdi > mdi:
            return "UPTREND"

        # Distribution: above EMA20 but momentum fading + RSI extended
        if ema5 > ema20 and slope < 47 and rsi > 50 and adx < 22:
            return "DISTRIBUTION"

        # Accumulation: below EMA20 + RSI neutral/oversold + low ADX
        if ema5 < ema20 and rsi < 50 and adx < 22:
            return "ACCUMULATION"

        # Exhaustion: elevated ADX but directional efficiency has collapsed
        # (no longer trending despite historical momentum).
        # Checked AFTER trend states so genuine uptrends aren't misclassified.
        if adx > 20 and eff < 0.28:
            return "EXHAUSTION"

        return "RANGE"

    # ── Step 2: 4H Regime score ──────────────────────────────────────────────

    def _step2_4h_regime(self, ind: Dict) -> float:
        score  = self._scale_score(ind.get("ema20_slope_score", 50), 30)
        score += self._scale_score(ind.get("adx_score",         50), 20)
        score += self._scale_score(ind.get("eff_score",         50), 20)
        score += self._scale_score(ind.get("atr_score",         50), 15)
        score += self._scale_score(ind.get("vol_score",         50), 15)
        return float(np.clip(score, 0, 100))

    # ── Step 3: Global gates ─────────────────────────────────────────────────

    def _check_global_gates(self) -> bool:
        if self.position_open:
            return False

        # [FIX 9] datetime comparison — use bar time in backtest, wall-clock in live
        _now = self._bar_now or datetime.datetime.now()

        # Post-restart warmup: wait for indicators/ATR history to stabilize
        if self._startup_unblock_at and _now < self._startup_unblock_at:
            remaining = int((self._startup_unblock_at - _now).total_seconds() / 60)
            self._log_event(f"WARMUP: {remaining}m remaining — no new entries", level="debug")
            return False

        if self.cooldown_until and _now < self.cooldown_until:
            self.state = "COOLDOWN"
            return False

        if self.daily_pnl_pct <= self.daily_loss_limit_pct:
            self.state = "BLOCKED"
            self._log_event(f"BLOCKED: daily PnL {self.daily_pnl_pct:.2f}% hit loss limit", level="warning")
            return False

        if self.daily_pnl_pct >= self.daily_profit_limit_pct:
            self.state = "BLOCKED"
            self._log_event(f"BLOCKED: daily PnL {self.daily_pnl_pct:.2f}% hit profit limit", level="warning")
            return False

        # MODULE 2: regime gate — ทุก state เทรดได้ แต่ต้องผ่าน threshold ต่างกัน
        if self.current_market_state not in _TRADEABLE_STATES:
            self._log_event(
                f"SKIP: unknown market_state={self.current_market_state}",
                level="debug",
            )
            return False

        # Direction compatibility with regime (DISTRIBUTION=SHORT only, ACCUMULATION=LONG only)
        if self.direction_focus is not None:
            rt = REGIME_THRESHOLDS.get(self.current_market_state, {})
            if self.direction_focus == "LONG" and not rt.get("allow_long", True):
                self._log_event(
                    f"SKIP: {self.current_market_state} blocks LONG entries",
                    level="debug",
                )
                return False
            if self.direction_focus == "SHORT" and not rt.get("allow_short", True):
                self._log_event(
                    f"SKIP: {self.current_market_state} blocks SHORT entries",
                    level="debug",
                )
                return False

        return True

    # ── Step 4: Layer filtering ──────────────────────────────────────────────

    def _layer_filtering(self, direction: str, ind: Dict,
                         market_state: str, vol_state: str) -> bool:
        weights = self.adaptive_engine.get_dynamic_weights(market_state)

        layer_score = 0.0
        for key in ["trend", "momentum", "volume", "pattern", "structure",
                    "atr", "rsi", "sweep", "divergence", "ema", "vwap", "macd"]:
            layer_score += self._scale_score(
                ind.get(f"{key}_score", 0), weights.get(key, 0)
            )

        multiplier = self.adaptive_engine.get_regime_multiplier(self.regime_score)
        layer_score *= multiplier

        thresholds = self.adaptive_engine.get_layer_thresholds(vol_state, market_state)
        return layer_score >= thresholds["L1_PASS"]

    # ── [FIX 17] Step 3.5: 1H trend filter — กรองทิศทางก่อนเข้าไม้จาก 15M ──────

    def _check_htf_trend_filter(self, direction: str, ind_1h: Dict) -> bool:
        """
        ยืนยันว่าแนวโน้ม 1H ไปทางเดียวกับสัญญาณที่เจอบน 15M ก่อนอนุญาตเข้าไม้
        (4H ใช้กำหนด market_state/regime ทั้งระบบอยู่แล้วใน Step1/Step2 — ส่วนนี้
        เป็นตัวกรองชั้นกลางเพิ่มเติมระหว่าง 4H กับ entry timeframe 15M)

        เกณฑ์: ADX 1H ต้องมีความแรงพอสมควร (>=15) และโครงสร้าง EMA5/EMA20 บน 1H
        ต้องสอดคล้องกับทิศทางที่จะเข้า — ถ้า 1H ไม่มีตัวบอกทิศทางชัดเจน ไม่อนุญาตเข้า
        ปรับเกณฑ์นี้ได้ตามสไตล์เทรดของคุณ (เช่น เพิ่มเช็ค trend_score หรือ adx_score ของ 1H)
        """
        adx_1h  = ind_1h.get("adx", 0)
        ema5_1h  = ind_1h.get("ema5")
        ema20_1h = ind_1h.get("ema20")

        # MODULE 2: regime-adaptive ADX minimum on 1H
        rt = REGIME_THRESHOLDS.get(self.current_market_state, REGIME_THRESHOLDS["UPTREND"])
        adx_min = rt["adx_1h_min"]

        if adx_1h < adx_min or ema5_1h is None or ema20_1h is None:
            return False

        # Allow EMA5 within 0.3% of EMA20 — catches turning points before full crossover.
        # Works for both trend-follow and reversal states; tested: strict crossover
        # for reversals reduced trades AND profits without improving WR.
        tolerance = ema20_1h * 0.003
        if direction == "LONG":
            return ema5_1h >= ema20_1h - tolerance
        else:
            return ema5_1h <= ema20_1h + tolerance

    # ── Step 5: Risk engine + sizing ─────────────────────────────────────────

    def _step5_risk_engine(self, candle: Dict, direction: str, ind: Dict):
        entry_price = float(candle.get("close", 0))

        if direction == "LONG":
            pattern_sl = candle.get("pattern_low", entry_price * 0.99)
        else:
            pattern_sl = candle.get("pattern_high", entry_price * 1.01)

        sl_dist = abs(entry_price - float(pattern_sl))
        if sl_dist < 1e-8:
            sl_dist = entry_price * 0.01

        # min_sl_pct floor (2.0%): widen both SL PRICE and sl_dist when the
        # pattern stop is too tight.  Prevents noise-stops AND oversized positions.
        # Matches the 2.0% floor applied by BacktestEngine.cfg.min_sl_pct.
        min_sl_dist = entry_price * 0.020
        if sl_dist < min_sl_dist:
            sl_dist = min_sl_dist
            pattern_sl = entry_price - sl_dist if direction == "LONG" else entry_price + sl_dist

        # [FIX] win streak risk reduction
        risk_pct = self.base_risk_pct
        if self.win_streak >= 5:
            risk_pct *= 0.80
            self._log_event(f"Win streak {self.win_streak} → risk reduced to {risk_pct:.2%}")

        risk_amount = self.account_balance * risk_pct
        position_size = risk_amount / sl_dist

        mult = 1 if direction == "LONG" else -1
        # TP1: 0.7R → close 50% + move SL to breakeven
        # TP2: 2.0R → close remaining 50% (raised from 1.5R for better expectancy)
        tp1 = entry_price + sl_dist * 0.7 * mult
        tp2 = entry_price + sl_dist * 2.0 * mult

        self.current_trade = {
            "direction":            direction,
            "entry":                entry_price,
            "sl":                   float(pattern_sl),
            "sl_dist":              sl_dist,
            "tp1":                  tp1,
            "tp2":                  tp2,
            "size":                 position_size,
            "tp1_hit":              False,
            "tp2_hit":              False,
            "break_even_triggered": False,
            "status":               "OPEN",
            "entry_time":           datetime.datetime.now(),
            "atr_at_entry":         ind.get("atr", sl_dist),
            "realized_pnl":         0.0,
            "remaining_size":       position_size,
            "exit_price":           None,
            "final_rr":             None,
            "mae":                  0.0,   # max adverse excursion
            "mfe":                  0.0,   # max favorable excursion
        }
        self.position_open = True
        self._position_entry_bar = self._bar_count
        self.order_status  = "OPEN"

        self._send_order("OPEN_LONG" if direction == "LONG" else "OPEN_SHORT", {
            "entry":     entry_price,
            "sl":        float(pattern_sl),
            "tp1":       tp1, "tp2": tp2,
            "size":      position_size,
        })

    # ── Step 6: Position management ──────────────────────────────────────────

    def _manage_open_position(self, current_price: float,
                              ind: Dict) -> str:
        t = self.current_trade
        direction = t["direction"]
        sl_dist = t["sl_dist"]
        dir_mult = 1 if direction == "LONG" else -1
        current_r = ((current_price - t["entry"]) * dir_mult) / max(sl_dist, 1e-9)

        # Track MAE / MFE
        t["mae"] = min(t["mae"], current_r)
        t["mfe"] = max(t["mfe"], current_r)

        # ── [FIX 7] Emergency exit conditions ─────────────────────────────
        emergency_signals = [
            ind.get("opposite_choch"),
            ind.get("atr_collapse"),
            ind.get("momentum_collapse"),
            ind.get("volume_collapse"),
            ind.get("invalid_structure"),
        ]
        if any(emergency_signals):
            self._close_position("EMERGENCY_EXIT", current_price, 1.0, ind)
            return "EXITING"

        # ── Partial TP — [FIX 16] เหลือแค่ TP1/TP2 ──────────────────────────
        # TP1 = scale-out ตาม tp1_close_pct แล้วขยับ SL ไป breakeven ทันที
        # TP2 = ปิดที่เหลือทั้งหมด (full exit) — ไม่มี TP3/runner เลย R สูงๆ อีกต่อไป
        if direction == "LONG":
            tp1_hit_cond = current_price >= t["tp1"]
            tp2_hit_cond = current_price >= t["tp2"]
        else:
            tp1_hit_cond = current_price <= t["tp1"]
            tp2_hit_cond = current_price <= t["tp2"]

        # TP1 — close ตาม tp1_close_pct + [FIX 16] ขยับ SL ไป breakeven ทันที
        if tp1_hit_cond and not t["tp1_hit"]:
            self._close_position("PARTIAL_TP1", t["tp1"], self.tp1_close_pct, ind)
            t["tp1_hit"] = True
            if not t["break_even_triggered"]:
                t["sl"] = t["entry"]
                t["break_even_triggered"] = True
                self._log_event(
                    f"TP1 hit @ {t['tp1']:.2f} → ปิด {self.tp1_close_pct:.0%} "
                    f"+ ขยับ SL ไป breakeven ({t['entry']:.2f})"
                )
            self.state = "PARTIAL_EXIT"

        # TP2 — ปิดที่เหลือทั้งหมด (full exit)
        if tp2_hit_cond and not t["tp2_hit"]:
            self._close_position("FULL_TP2", t["tp2"], 1.0, ind)
            t["tp2_hit"] = True
            return "EXITING"

        # ── [FIX 1] Health-based position management ──────────────────────
        # Before TP1: health actions gated — TP1 must be hit first.
        # In Compression/Sideway states ADX is low by design, which systematically
        # pulls health to 40-60 and triggers premature break-even before the trade
        # has any chance to reach TP1. Let the original SL do its job pre-TP1.
        # After TP1: full health management watches the remaining 50% position.
        tp1_hit = t.get("tp1_hit", False)
        health = self.health_calc.calculate(ind, t, current_price)

        if health >= 80:
            self._log_event(
                f"Health {health:.0f} (≥80) → HOLD: ปล่อย position วิ่งต่อ "
                f"(current_r={current_r:.2f})",
                level="debug",
            )

        elif health >= 60 or t["break_even_triggered"]:
            # ATR trailing stop (runs both pre and post TP1 — only tightens SL)
            atr = ind.get("atr", sl_dist)
            atr_trail = atr * 2
            if direction == "LONG":
                new_sl = current_price - atr_trail
                t["sl"] = max(t["sl"], new_sl)
            else:
                new_sl = current_price + atr_trail
                t["sl"] = min(t["sl"], new_sl)

        elif health >= 40:
            # Force break-even only AFTER TP1 has been hit — prevents Compression-state
            # false-positive health drops from cutting trades before they develop
            if not t["break_even_triggered"] and tp1_hit:
                t["sl"] = t["entry"]
                t["break_even_triggered"] = True
                self._log_event(f"Health {health:.0f} → forced break-even (post-TP1)")

        elif health >= 20:
            # Reduce position size — only act on the remaining portion post-TP1
            if t["remaining_size"] > 0 and tp1_hit:
                self._close_position("HEALTH_REDUCE", current_price, 0.50, ind)
                self.state = "PARTIAL_EXIT"

        else:
            # Emergency exit regardless of TP1 state — health is critically low
            self._close_position("POOR_HEALTH_EXIT", current_price, 1.0, ind)
            return "EXITING"

        # ── SL hit check ──────────────────────────────────────────────────
        if direction == "LONG" and current_price <= t["sl"]:
            self._close_position("SL_HIT", t["sl"], 1.0, ind)
            return "EXITING"
        if direction == "SHORT" and current_price >= t["sl"]:
            self._close_position("SL_HIT", t["sl"], 1.0, ind)
            return "EXITING"

        return self.state

    # ── [FIX 2] Close position — คำนวณ PnL จริง ─────────────────────────────

    def _close_position(self, reason: str, price: float,
                        portion: float, ind: Dict):
        """
        [FIX 2] บันทึก PnL จริง อัปเดต current_trade และส่งคำสั่ง

        portion: 0.0–1.0 (1.0 = ปิดทั้งหมด)
        """
        t = self.current_trade
        if not t or t.get("status") != "OPEN":
            return

        direction   = t["direction"]
        entry_price = t["entry"]
        size        = t["size"]
        remaining   = t.get("remaining_size", size)

        close_size = remaining * min(max(float(portion), 0.0), 1.0)
        if close_size <= 0:
            return

        # คำนวณ PnL
        if direction == "LONG":
            pnl_per_unit = float(price) - entry_price
        else:
            pnl_per_unit = entry_price - float(price)

        pnl = pnl_per_unit * close_size

        # อัปเดต state
        t["realized_pnl"] = t.get("realized_pnl", 0.0) + pnl
        t["remaining_size"] = remaining - close_size
        t["exit_price"] = float(price)

        if t["sl_dist"] > 0:
            t["final_rr"] = pnl_per_unit / t["sl_dist"]

        self._send_order("CLOSE_PARTIAL" if portion < 1.0 else "CLOSE_FULL", {
            "reason":    reason,
            "price":     float(price),
            "size":      close_size,
            "pnl":       pnl,
            "direction": direction,
        })

        self._log_event(
            f"CLOSE {reason} | {direction} | price={price:.2f} "
            f"| size={close_size:.4f} | pnl={pnl:+.2f}"
        )

        # ถ้าปิดหมดแล้ว
        if portion >= 1.0 or t["remaining_size"] <= 1e-9:
            t["status"] = "CLOSED"
            self.position_open = False
            self.order_status  = "CLOSED"

    # ── Step 7: Journal + learning ────────────────────────────────────────────

    def _log_trade(self, result: str, ind: Dict, extras: Dict):
        """
        [FIX 10] บันทึก trade พร้อม active conditions สำหรับ per-condition learning
        """
        t = self.current_trade
        pnl = t.get("realized_pnl", 0.0)

        # Active conditions ณ เวลา entry (จาก indicator snapshot)
        active_conditions = {
            "trend":      ind.get("trend_score", 0) > 60,
            "momentum":   ind.get("momentum_score", 0) > 60,
            "volume":     ind.get("volume_score", 0) > 60,
            "pattern":    ind.get("pattern_score", 0) > 60,
            "structure":  ind.get("structure_score", 0) > 60,
            "atr":        ind.get("atr_score", 0) > 60,
            "rsi":        ind.get("rsi_score", 0) > 60,
            "sweep":      ind.get("sweep_score", 0) > 60,
            "divergence": ind.get("divergence_score", 0) > 60,
            "ema":        ind.get("ema_score", 0) > 60,
            "vwap":       ind.get("vwap_score", 0) > 60,
            "macd":       ind.get("macd_score", 0) > 60,
        }

        # MODULE 3: compute R-multiples for expectancy learning
        _entry  = t.get("entry", 0.0)
        _sl     = t.get("sl", _entry)
        _exit   = t.get("exit_price", _entry)
        _sl_d   = max(abs(_entry - _sl), 1e-8)
        _d_mult = 1 if t.get("direction") == "LONG" else -1
        _realized_r = _d_mult * (_exit - _entry) / _sl_d
        _win_r  = max(_realized_r, 0.0)
        _loss_r = max(-_realized_r, 0.0)

        _vol_state = self.adaptive_engine.get_atr_volatility_state(
            ind.get("atr", 0), self.atr_history)
        _holding_bars = self._bar_count - self._position_entry_bar

        entry = {
            "symbol":           extras.get("symbol", "BTCUSDT"),
            "timeframe":        "15M",
            "direction":        t.get("direction"),
            "entry":            t.get("entry"),
            "exit":             t.get("exit_price"),
            "sl":               t.get("sl"),
            "tp1":              t.get("tp1"),
            "tp2":              t.get("tp2"),
            "rr":               t.get("final_rr"),
            "win_loss":         result,
            "pnl":              pnl,
            "win_r":            round(_win_r, 3),
            "loss_r":           round(_loss_r, 3),
            "mae":              t.get("mae"),
            "mfe":              t.get("mfe"),
            "holding_bars":     _holding_bars,
            "hold_time_sec":    (datetime.datetime.now() - t.get("entry_time", datetime.datetime.now())).total_seconds(),
            "atr":              ind.get("atr"),
            "adx":              ind.get("adx"),
            "rsi":              ind.get("rsi"),
            "ema":              ind.get("ema5"),
            "macd":             ind.get("macd"),
            "market_state":     self.current_market_state,
            "entry_type":       _REGIME_ENTRY_TYPE.get(self.current_market_state, "trend_follow"),
            "atr_percentile":   _vol_state,
            "hour_utc":         (self._bar_now or datetime.datetime.now()).hour,
            "volatility_state": _vol_state,
            "regime_score":     self.regime_score,
            "session":          extras.get("session"),
            "funding":          extras.get("funding_rate"),
            "open_interest":    extras.get("oi"),
            "pattern":          ind.get("pattern"),
            "entry_reason":     "Strategy Trigger",
            "exit_reason":      result,
            "active_conditions": active_conditions,
        }
        self.trade_journal.append(entry)

        # MODULE 3: record to learning engine with R-multiples
        self.adaptive_engine.record_trade_result(
            self.current_market_state,
            win=(result == "WIN"),
            active_conditions=active_conditions,
            win_r=_win_r,
            loss_r=_loss_r,
        )

        # Update streaks + daily PnL
        self.daily_pnl_pct += (pnl / max(self.account_balance, 1)) * 100
        self.account_balance += pnl

        if result == "WIN":
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0
            if self.loss_streak >= self.max_loss_streak:
                # [FIX 9] datetime cooldown — bar time in backtest, wall-clock in live
                _now = self._bar_now or datetime.datetime.now()
                self.cooldown_until = (_now
                                       + datetime.timedelta(minutes=self.cooldown_minutes))
                self.state = "COOLDOWN"
                self._log_event(
                    f"Loss streak {self.loss_streak} → COOLDOWN until "
                    f"{self.cooldown_until.strftime('%H:%M:%S')}",
                    level="warning",
                )

        # Learning every 100 trades
        if len(self.trade_journal) % 100 == 0:
            self.adaptive_engine.adjust_weights_from_learning(self.trade_journal[-100:])
            self._log_event("Learning engine updated weights from last 100 trades")

        self._log_event(
            f"TRADE CLOSED | {result} | PnL={pnl:+.2f} "
            f"| Balance={self.account_balance:.2f} "
            f"| WinStreak={self.win_streak} LossStreak={self.loss_streak}"
        )

    # ── Daily reset ───────────────────────────────────────────────────────────

    def _check_daily_reset(self, bar_dt: Optional[datetime.datetime] = None):
        today = (bar_dt.date() if bar_dt else datetime.date.today())
        if self.trading_date != today:
            self.trading_date = today
            self.daily_pnl_pct = 0.0
            if self.state == "BLOCKED":
                self.state = "SCANNING"
                self._log_event("New trading day — BLOCKED state reset")

    # ── Core tick engine ──────────────────────────────────────────────────────

    def on_tick(self, candle_15m: Dict, candle_1h: Dict, candle_4h: Dict,
                ind_15m: Dict, ind_1h: Dict, ind_4h: Dict,
                extras: Dict, current_price: float,
                bar_dt: Optional[datetime.datetime] = None):
        """
        เรียกต่อ closed candle หนึ่งครั้ง (ตาม timeframe เข้าออเดอร์ คือทุกแท่ง 15M ปิด)

        bar_dt : datetime ของแท่ง 15M (ส่งมาจาก backtest หรือ live runner)
                 ถ้า None จะใช้ datetime.now() — ถูกต้องสำหรับ live trading
                 ส่ง bar_dt จาก backtest เพื่อให้ cooldown/daily-reset อิง bar time
                 แทน wall-clock time (กันกรณี backtest ประมวลผลเร็วกว่า 30 นาทีจริง)
        """
        now = bar_dt or datetime.datetime.now()
        self._bar_now = now  # shared across all sub-methods this tick
        self._bar_count += 1

        # Set startup warmup window on very first tick
        if self._startup_unblock_at is None and self.startup_warmup_minutes > 0:
            self._startup_unblock_at = now + datetime.timedelta(minutes=self.startup_warmup_minutes)
            self._log_event(
                f"POST-RESTART WARMUP: blocking new entries until "
                f"{self._startup_unblock_at.strftime('%H:%M UTC')} "
                f"({self.startup_warmup_minutes}m)",
                level="warning",
            )

        self._check_daily_reset(bar_dt)

        # อัปเดต ATR history — ใช้ 4H เพื่อความนิ่งของ volatility regime classification
        # (ไม่ใช้ ATR ของ 15M ตรงนี้ เพราะจะ noisy เกินไปสำหรับ threshold ระดับ regime;
        #  ส่วน SL distance/position sizing ใน _step5_risk_engine ใช้ pattern/ATR ของ 15M เอง)
        atr_4h = ind_4h.get("atr", 0)
        if atr_4h > 0:
            self.atr_history.append(atr_4h)
            if len(self.atr_history) > 200:
                self.atr_history.pop(0)

        # อัปเดต market state จาก 4H (macro regime)
        self.current_market_state = self._step1_market_state_engine(ind_4h)
        self.regime_score          = self._step2_4h_regime(ind_4h)
        vol_state                  = self.adaptive_engine.get_atr_volatility_state(
                                         atr_4h, self.atr_history)

        # [FIX 5] State machine loop — guard ป้องกัน infinite loop
        self._tick_depth = 0
        state_changed = True

        while state_changed and self._tick_depth < self._max_tick_depth:
            state_changed = False
            self._tick_depth += 1

            # ── BLOCKED / COOLDOWN ─────────────────────────────────────────
            if self.state in ("BLOCKED", "COOLDOWN"):
                # [FIX 9] datetime comparison — use bar time in backtest, wall-clock in live
                if (self.state == "COOLDOWN" and
                        (self.cooldown_until is None or
                         now >= self.cooldown_until)):
                    self.state = "SCANNING"
                    self._log_event("Cooldown expired → SCANNING")
                    state_changed = True

            # ── SCANNING ───────────────────────────────────────────────────
            elif self.state == "SCANNING":
                # [FIX] Reset direction_focus on each SCANNING pass.
                # direction_focus persists from FILTERING and is never cleared
                # when a trade exits, a WAIT_CONFIRM times out, or the regime
                # changes.  If it stays as "LONG" when the new regime is
                # DISTRIBUTION (allow_long=False), _check_global_gates()
                # permanently returns False — bot stuck in SCANNING forever.
                self.direction_focus = None
                if self._check_global_gates():
                    self.state = "FILTERING"
                    state_changed = True

            # ── FILTERING ──────────────────────────────────────────────────
            # [FIX 17] scoring ใช้ ind_15m (entry timeframe) แต่ต้องผ่าน 1H trend
            # filter ก่อน ไม่ใช่แค่ดู 15M เพียวๆ — กันสัญญาณสวนเทรนด์ใหญ่
            elif self.state == "FILTERING":
                # Score both directions, pick the highest-scoring one that passes all filters
                best_dir: Optional[str] = None
                best_score: float = -1.0
                weights = self.adaptive_engine.get_dynamic_weights(self.current_market_state)
                thresholds = self.adaptive_engine.get_layer_thresholds(vol_state, self.current_market_state)
                multiplier = self.adaptive_engine.get_regime_multiplier(self.regime_score)
                rt_current = REGIME_THRESHOLDS.get(self.current_market_state, {})
                for direction in ("LONG", "SHORT"):
                    # [FIX] Enforce regime direction gates (allow_long/allow_short)
                    # These were dead code in _check_global_gates (direction_focus=None there)
                    if direction == "LONG" and not rt_current.get("allow_long", True):
                        continue
                    if direction == "SHORT" and not rt_current.get("allow_short", True):
                        continue
                    if not self._check_htf_trend_filter(direction, ind_1h):
                        continue
                    # [FIX] Direction-aware scoring: SHORT in bearish market should score HIGH,
                    # not LOW.  Invert LONG-biased indicator scores for SHORT direction so that
                    # bearish conditions (structure=15, ema_score<50, momentum<50, etc.) become
                    # high scores that properly reflect SHORT-favourable conditions.
                    # Neutral indicators (volume, atr, rsi, sweep, divergence, pattern, trend)
                    # are unchanged — trend_score is already direction-agnostic (high for any
                    # strong trend regardless of direction).
                    if direction == "SHORT":
                        dir_ind = {**ind_15m,
                                   **{f"{k}_score": 100.0 - ind_15m.get(f"{k}_score", 50.0)
                                      for k in _DIR_INVERT_KEYS}}
                    else:
                        dir_ind = ind_15m
                    raw_score = sum(
                        self._scale_score(dir_ind.get(f"{k}_score", 0), weights.get(k, 0))
                        for k in weights
                    ) * multiplier
                    if raw_score >= thresholds["L1_PASS"] and raw_score > best_score:
                        best_score = raw_score
                        best_dir = direction

                if best_dir is not None:
                    self.direction_focus   = best_dir
                    self.state             = "WAIT_CONFIRM"
                    self.bars_since_trigger = 0
                    self._log_event(
                        f"Signal found: {best_dir} in {self.current_market_state} "
                        f"score={best_score:.1f} (1H trend filter ผ่าน)"
                    )
                    state_changed = True
                else:
                    self.state = "SCANNING"

            # ── WAIT_CONFIRM ───────────────────────────────────────────────
            elif self.state == "WAIT_CONFIRM":
                if self.bars_since_trigger > 3:  # 3 bars = 45 min บน 15M (ลด miss window)
                    self._log_event("WAIT_CONFIRM timeout → back to SCANNING")
                    self.state    = "SCANNING"
                    state_changed = True
                    continue

                # ตรวจสอบ trigger condition จาก indicator 15M (entry timeframe)
                confirmed = self._check_entry_trigger(ind_15m)
                if confirmed:
                    self.state = "PENDING_ORDER"
                    state_changed = True
                else:
                    self.bars_since_trigger += 1

            # ── PENDING_ORDER ──────────────────────────────────────────────
            elif self.state == "PENDING_ORDER":
                self._step5_risk_engine(candle_15m, self.direction_focus, ind_15m)
                self.state    = "IN_POSITION"
                state_changed = False  # หยุด loop — รอ tick ถัดไปตรวจ position

            # ── IN_POSITION / PARTIAL_EXIT / TRAILING ─────────────────────
            elif self.state in ("IN_POSITION", "PARTIAL_EXIT", "TRAILING"):
                new_state = self._manage_open_position(current_price, ind_15m)
                if new_state == "EXITING":
                    self.state = "EXITING"
                    state_changed = True
                elif new_state != self.state:
                    self.state = new_state

            # ── EXITING ────────────────────────────────────────────────────
            elif self.state == "EXITING":
                t     = self.current_trade
                pnl   = t.get("realized_pnl", 0.0)
                result = "WIN" if pnl > 0 else "LOSS"
                self._log_trade(result, ind_15m, extras)
                if self.state not in ("COOLDOWN", "BLOCKED"):
                    self.state = "SCANNING"

            # ── [FIX 8] RECOVERY ───────────────────────────────────────────
            elif self.state == "RECOVERY":
                # Recovery mode: รอ 1 trade ที่ risk ลดลง 50% ก่อน resume full risk
                if not self.position_open:
                    self._log_event("RECOVERY mode: waiting for next setup with half-risk")
                    if self._check_global_gates():
                        self._enter_recovery_trade(candle_15m, ind_15m, ind_1h, vol_state)

            # ── ERROR ──────────────────────────────────────────────────────
            elif self.state == "ERROR":
                self._log_event("Bot in ERROR state — manual intervention required", level="error")
                break

        # [FIX 13] persist state ทุก tick — กัน crash/restart เสีย state เกิน 1 tick
        # (เขียนแบบ atomic write ใน save_state แล้ว จึงไม่กระทบ performance loop มากนัก
        #  ถ้าต้องการลด I/O สามารถปรับให้ save แค่ตอน state เปลี่ยนแทนได้)
        self.save_state(self._state_file)

    def _check_entry_trigger(self, ind: Dict) -> bool:
        """
        ตรวจสอบ entry trigger condition จาก indicator 15M (entry timeframe)
        momentum_score: 50 + direction * strength  (>50 = bullish, <50 = bearish)
        """
        direction = self.direction_focus
        if direction is None:
            return False

        rsi      = ind.get("rsi", 50)
        momentum = ind.get("momentum_score", 50)

        # MODULE 2: regime-specific RSI entry gates
        rt = REGIME_THRESHOLDS.get(self.current_market_state, REGIME_THRESHOLDS["UPTREND"])

        if direction == "LONG":
            lo, hi = rt["rsi_long"]
            return lo < rsi < hi and momentum > 50
        else:
            lo, hi = rt["rsi_short"]
            return lo < rsi < hi and momentum < 50

    def _enter_recovery_trade(self, candle: Dict, ind: Dict, ind_1h: Dict, vol_state: str):
        """[FIX 8] เปิด trade ด้วย risk 50% ใน recovery mode (ยังผ่าน 1H trend filter เหมือนปกติ)"""
        original_risk = self.base_risk_pct
        self.base_risk_pct *= 0.5

        for direction in ("LONG", "SHORT"):
            if not self._check_htf_trend_filter(direction, ind_1h):
                continue
            if self._layer_filtering(direction, ind, self.current_market_state, vol_state):
                self._step5_risk_engine(candle, direction, ind)
                self.state = "IN_POSITION"
                self._log_event(f"RECOVERY trade opened: {direction} at half-risk")
                break

        self.base_risk_pct = original_risk

    # ── Public API ────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Summary ของ bot state ปัจจุบัน"""
        _now = self._bar_now or datetime.datetime.now()
        warmup_remaining = 0
        if self._startup_unblock_at and _now < self._startup_unblock_at:
            warmup_remaining = int((self._startup_unblock_at - _now).total_seconds() / 60)
        return {
            "state":              self.state,
            "position_open":      self.position_open,
            "direction":          self.current_trade.get("direction") if self.position_open else None,
            "entry":              self.current_trade.get("entry") if self.position_open else None,
            "realized_pnl":       self.current_trade.get("realized_pnl", 0),
            "account_balance":    self.account_balance,
            "daily_pnl_pct":      self.daily_pnl_pct,
            "loss_streak":        self.loss_streak,
            "win_streak":         self.win_streak,
            "total_trades":       len(self.trade_journal),
            "market_state":       self.current_market_state,
            "regime_score":       self.regime_score,
            "cooldown_until":     self.cooldown_until.isoformat() if self.cooldown_until else None,
            "warmup_remaining_m": warmup_remaining,
            "recent_log":         list(self._log[-20:]),
        }

    def get_performance_summary(self) -> Dict:
        """สรุปผลการเทรดจาก journal"""
        if not self.trade_journal:
            return {"message": "No trades yet"}

        wins   = [t for t in self.trade_journal if t["win_loss"] == "WIN"]
        losses = [t for t in self.trade_journal if t["win_loss"] == "LOSS"]
        pnls   = [t["pnl"] for t in self.trade_journal if t.get("pnl") is not None]

        by_state = {}
        for t in self.trade_journal:
            s = t.get("market_state", "Unknown")
            by_state.setdefault(s, {"wins": 0, "total": 0, "pnl": 0.0})
            by_state[s]["total"] += 1
            by_state[s]["pnl"]   += t.get("pnl", 0)
            if t["win_loss"] == "WIN":
                by_state[s]["wins"] += 1

        return {
            "total_trades":  len(self.trade_journal),
            "win_rate":      len(wins) / len(self.trade_journal),
            "net_pnl":       sum(pnls),
            "avg_win":       np.mean([t["pnl"] for t in wins])   if wins   else 0,
            "avg_loss":      np.mean([t["pnl"] for t in losses]) if losses else 0,
            "profit_factor": (sum(t["pnl"] for t in wins) /
                              abs(sum(t["pnl"] for t in losses)))
                             if losses else float("inf"),
            "by_market_state": {
                s: {**v, "win_rate": v["wins"] / max(v["total"], 1)}
                for s, v in by_state.items()
            },
        }

    def force_state(self, new_state: str):
        """Manual override สำหรับ emergency หรือ testing"""
        if new_state in self.STATES:
            self._log_event(f"Force state: {self.state} → {new_state}")
            self.state = new_state
        else:
            raise ValueError(f"Unknown state: {new_state}. Valid: {self.STATES}")

    # ── [FIX 13] State persistence — รองรับการรีสตาร์ตบอท ───────────────────
    #
    # บอทเทรดจริงต้องรันยาว ๆ บน VPS/server และอาจถูก restart (deploy ใหม่,
    # crash, server reboot) ระหว่างที่ "มี position เปิดอยู่" ถ้าไม่เก็บ state
    # บอทจะเริ่มจาก SCANNING ใหม่ทั้งที่ยังมี position ค้างอยู่จริงบน exchange
    # → เสี่ยงเปิด position ซ้ำ หรือไม่มีใครดูแล SL/TP/trailing ของ position เดิม

    DEFAULT_STATE_FILE = "bot_state.json"

    def save_state(self, path: str = DEFAULT_STATE_FILE):
        """บันทึก state ปัจจุบันลงไฟล์ JSON (เรียกทุกครั้งที่ state สำคัญเปลี่ยน)"""
        import os as _os
        if not path or path == _os.devnull:
            return  # backtest mode — ไม่ต้องเขียน disk
        snapshot = {
            "state":                self.state,
            "position_open":        self.position_open,
            "order_status":         self.order_status,
            "current_trade":        self.current_trade,
            "account_balance":      self.account_balance,
            "daily_pnl_pct":        self.daily_pnl_pct,
            "loss_streak":          self.loss_streak,
            "win_streak":           self.win_streak,
            "cooldown_until":       self.cooldown_until.isoformat() if self.cooldown_until else None,
            "trading_date":         self.trading_date.isoformat() if self.trading_date else None,
            "current_market_state": self.current_market_state,
            "regime_score":         self.regime_score,
            "direction_focus":      self.direction_focus,
            "bars_since_trigger":   self.bars_since_trigger,
            "atr_history":          self.atr_history[-200:],
            "adaptive_weights":     self.adaptive_engine.base_weights,
            "saved_at":             datetime.datetime.now().isoformat(),
        }
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, default=str, ensure_ascii=False)
            os.replace(tmp_path, path)  # atomic write — กัน state file พังถ้า crash กลางทาง
        except OSError as e:
            self._log_event(f"[ERROR] save_state failed: {e}", level="error")

    def load_state(self, path: str = DEFAULT_STATE_FILE) -> bool:
        """
        โหลด state จากไฟล์ JSON ตอนสตาร์ทบอท
        Return True ถ้าโหลดสำเร็จ, False ถ้าไม่มีไฟล์ (= สตาร์ทใหม่)
        """
        if not os.path.exists(path):
            self._log_event(f"No saved state at '{path}' — starting fresh")
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self._log_event(
                f"[ERROR] load_state failed to read '{path}': {e} — starting fresh instead",
                level="error",
            )
            return False

        self.state           = data.get("state", "SCANNING")
        self.position_open    = data.get("position_open", False)
        self.order_status     = data.get("order_status", "CLOSED")
        self.current_trade    = data.get("current_trade") or {}

        # entry_time ถูกเก็บเป็น ISO string ใน JSON ต้องแปลงกลับเป็น datetime
        entry_time = self.current_trade.get("entry_time")
        if isinstance(entry_time, str):
            try:
                self.current_trade["entry_time"] = datetime.datetime.fromisoformat(entry_time)
            except ValueError:
                self.current_trade["entry_time"] = datetime.datetime.now()

        self.account_balance  = data.get("account_balance", self.account_balance)
        self.daily_pnl_pct     = data.get("daily_pnl_pct", 0.0)
        self.loss_streak       = data.get("loss_streak", 0)
        self.win_streak         = data.get("win_streak", 0)

        cooldown = data.get("cooldown_until")
        self.cooldown_until = datetime.datetime.fromisoformat(cooldown) if cooldown else None

        trading_date = data.get("trading_date")
        self.trading_date = datetime.date.fromisoformat(trading_date) if trading_date else None

        self.current_market_state = data.get("current_market_state", "Sideway")
        self.regime_score          = data.get("regime_score", 0.0)
        self.direction_focus        = data.get("direction_focus")
        self.bars_since_trigger      = data.get("bars_since_trigger", 0)
        self.atr_history              = data.get("atr_history", [])

        weights = data.get("adaptive_weights")
        if weights:
            self.adaptive_engine.base_weights = weights

        self._log_event(
            f"State loaded from '{path}' | state={self.state} "
            f"| position_open={self.position_open} | balance={self.account_balance:.2f} "
            f"| saved_at={data.get('saved_at')}"
        )
        return True

    def reconcile_with_exchange(self, symbol: str, exchange_adapter) -> None:
        """
        [FIX 13] เทียบ local state (จาก load_state) กับ position จริงบน exchange
        ต้องเรียกทุกครั้งหลัง restart ก่อนเริ่ม on_tick — เพราะระหว่างบอท offline
        position อาจถูกปิดไปแล้ว (SL/TP บน exchange โดน trigger) หรือมี position
        ที่ exchange แต่ไฟล์ state เก่าเกินไป/หาย

        exchange_adapter ต้องมีเมธอด fetch_open_position(symbol) -> dict | None
        (ดู OKXExecutionAdapter ใน okx_exchange.py)
        """
        try:
            live_position = exchange_adapter.fetch_open_position(symbol)
        except Exception as e:
            self._log_event(
                f"[ERROR] reconcile_with_exchange: fetch_open_position failed: {e} "
                f"— เข้า ERROR state เพื่อให้ตรวจสอบ manual ก่อนเทรดต่อ",
                level="error",
            )
            self.state = "ERROR"
            return

        live_size = float((live_position or {}).get("contracts", 0) or 0)

        if self.position_open and live_size == 0:
            self._log_event(
                "[RECONCILE] local state มี position เปิดอยู่ แต่ exchange ไม่มี position จริง "
                "→ ถือว่าปิดไปแล้วระหว่างบอท offline, เคลียร์ local state และกลับไป SCANNING",
                level="warning",
            )
            self.position_open = False
            self.current_trade = {}
            if self.state not in ("BLOCKED", "COOLDOWN", "ERROR"):
                self.state = "SCANNING"

        elif not self.position_open and live_size != 0:
            direction = "LONG" if live_size > 0 else "SHORT"
            entry_price = float(
                live_position.get("entryPrice") or live_position.get("entryPx") or 0
            )
            self._log_event(
                f"[RECONCILE] พบ position จริงบน exchange ({direction} size={abs(live_size)}) "
                "ที่ local state ไม่รู้จัก (อาจเปิดด้วยมือ หรือ state file หาย) "
                "→ สร้าง trade record ใหม่จากข้อมูล exchange — "
                "SL/TP/health logic จะ baseline ใหม่จากราคาปัจจุบัน",
                level="warning",
            )
            self.current_trade = {
                "direction": direction, "entry": entry_price,
                "sl": entry_price * (0.99 if direction == "LONG" else 1.01),
                "sl_dist": max(entry_price * 0.01, 1e-9),
                "tp1": entry_price, "tp2": entry_price,
                "size": abs(live_size), "remaining_size": abs(live_size),
                "tp1_hit": False, "tp2_hit": False,
                "break_even_triggered": False, "status": "OPEN",
                "entry_time": datetime.datetime.now(),
                "atr_at_entry": 0.0, "realized_pnl": 0.0,
                "exit_price": None, "final_rr": None, "mae": 0.0, "mfe": 0.0,
                "reconciled_from_exchange": True,
            }
            self.position_open = True
            self.state = "IN_POSITION"

        else:
            self._log_event("[RECONCILE] local state ตรงกับ exchange แล้ว — ไม่ต้องแก้ไข")
