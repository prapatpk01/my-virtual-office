"""
Market Health Monitor — checks every 15 minutes, independent of strategy ticks.

Score bands (0-100):
  > 85   Bull        — full operation, all entries allowed
  50-84  Neutral     — hold existing, new entries allowed
  25-49  Weak        — block new entries; soft-close after 3 consecutive weak checks
   < 25  StrongWeak  — close ALL positions immediately

Scoring breakdown (max 100):
  EMA20(4H) > EMA50(4H)         +25   trend direction
  EMA50(4H) > EMA200(4H)        +15   medium-term bias (needs 200+ bars)
  ADX(4H) rising                +15   trend strengthening
  RSI(4H) in 40-70              +10   healthy momentum range
  MACD hist(4H) > 0             +15   momentum positive
  1D macro (EMA50>EMA200 or RSI>50) +20  macro backdrop
"""
import logging
import math

from .strategies.base import BaseStrategy

logger = logging.getLogger("health_monitor")

_INTERVAL_MS  = 15 * 60 * 1000   # 15 minutes
_DEFAULT_SCORE = 60               # returned when not enough data (neutral-safe)


class HealthMonitor:

    def __init__(self, weak_bars_confirm: int = 3):
        self._weak_bars_confirm = weak_bars_confirm
        self._weak_count  = 0
        self._last_ts_ms  = 0
        self._prev_action = "hold"

        self.last_score  = _DEFAULT_SCORE
        self.last_label  = "neutral"
        self.last_action = "hold"

    # ── Public ───────────────────────────────────────────────────────────────

    def should_check(self, now_ms: int) -> bool:
        return (now_ms - self._last_ts_ms) >= _INTERVAL_MS

    def update(self, mtf: dict, now_ms: int) -> dict:
        """
        Recompute health.  Call when should_check() returns True.

        Returns:
            {
              "score":   int (0-100),
              "label":   "bull" | "neutral" | "weak" | "strong_weak",
              "action":  "hold" | "block_buy" | "soft_close" | "hard_close",
              "changed": bool  (True when action differs from previous check),
            }
        """
        score          = self._compute(mtf)
        label, action  = self._classify(score)
        changed        = action != self._prev_action

        self.last_score  = score
        self.last_label  = label
        self.last_action = action
        self._last_ts_ms = now_ms
        self._prev_action = action

        logger.info(
            "[Health] score=%d  label=%-11s  action=%-10s  weak_streak=%d/%d%s",
            score, label, action,
            self._weak_count, self._weak_bars_confirm,
            "  ← CHANGED" if changed else "",
        )
        return {"score": score, "label": label, "action": action, "changed": changed}

    # ── Internal ─────────────────────────────────────────────────────────────

    def _classify(self, score: int) -> tuple:
        if score > 85:
            self._weak_count = 0
            return "bull", "hold"
        if score >= 50:
            self._weak_count = 0
            return "neutral", "hold"
        if score >= 25:
            self._weak_count += 1
            if self._weak_count >= self._weak_bars_confirm:
                return "weak", "soft_close"
            return "weak", "block_buy"
        # < 25
        self._weak_count = 0
        return "strong_weak", "hard_close"

    def _compute(self, mtf: dict) -> int:
        c4h = mtf.get("4h", [])
        c1d = mtf.get("1d", [])

        if len(c4h) < 30:
            return _DEFAULT_SCORE   # not enough 4H bars → neutral

        closes_4h = [float(c.close) for c in c4h]
        n = len(c4h) - 1
        score = 0

        # ── EMA20 > EMA50 (4H) → +25 ─────────────────────────────────────
        ema20 = BaseStrategy.ema(closes_4h, 20)
        ema50 = BaseStrategy.ema(closes_4h, 50)
        e20, e50 = float(ema20[n]), float(ema50[n])
        if not (math.isnan(e20) or math.isnan(e50)) and e20 > e50:
            score += 25

        # ── EMA50 > EMA200 (4H) → +15 (needs ≥205 bars) ─────────────────
        if len(c4h) >= 205:
            e200 = float(BaseStrategy.ema(closes_4h, 200)[n])
            if not (math.isnan(e50) or math.isnan(e200)) and e50 > e200:
                score += 15

        # ── ADX(4H) rising → +15 ─────────────────────────────────────────
        adx_arr, _, _ = BaseStrategy.adx(c4h, 14)
        adx_v    = float(adx_arr[n])
        adx_prev = float(adx_arr[n - 1]) if n >= 1 else adx_v
        if not (math.isnan(adx_v) or math.isnan(adx_prev)) and adx_v > adx_prev:
            score += 15

        # ── RSI(4H) 40-70 → +10 ──────────────────────────────────────────
        if len(closes_4h) >= 20:
            rsi_v = float(BaseStrategy.rsi(closes_4h, 14)[n])
            if not math.isnan(rsi_v) and 40.0 <= rsi_v <= 70.0:
                score += 10

        # ── MACD hist(4H) > 0 → +15 ──────────────────────────────────────
        if len(closes_4h) >= 35:
            _, _, mh = BaseStrategy.macd(closes_4h, 12, 26, 9)
            mhv = float(mh[n]) if n < len(mh) else float("nan")
            if not math.isnan(mhv) and mhv > 0:
                score += 15

        # ── 1D macro → +20 ───────────────────────────────────────────────
        if len(c1d) >= 55:
            closes_1d = [float(c.close) for c in c1d]
            d         = len(c1d) - 1
            rsi_1d    = float(BaseStrategy.rsi(closes_1d, 14)[d])
            e50_1d    = float(BaseStrategy.ema(closes_1d, 50)[d])
            macro_ok  = not math.isnan(rsi_1d) and rsi_1d > 50
            if not macro_ok and len(c1d) >= 205:
                e200_1d = float(BaseStrategy.ema(closes_1d, 200)[d])
                if not (math.isnan(e50_1d) or math.isnan(e200_1d)) and e50_1d > e200_1d:
                    macro_ok = True
            if macro_ok:
                score += 20

        return min(score, 100)

    # ── Description helpers ───────────────────────────────────────────────────

    @staticmethod
    def label_emoji(label: str) -> str:
        return {"bull": "🟢", "neutral": "🟡", "weak": "🟠", "strong_weak": "🔴"}.get(label, "⚪")

    def summary(self) -> str:
        emoji = self.label_emoji(self.last_label)
        return (f"{emoji} Health {self.last_score}/100 "
                f"({self.last_label}) → {self.last_action}")
