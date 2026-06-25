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

Spike protection (runs every tick, 30-min cooldown):
  C1  TR of last 2 bars > 2.5×ATR14(30m)  — extreme candle / V-spike
  C2  2-bar cumulative drop  > 2.0×ATR14   — rapid cascade fall
  C3  3-bar cumulative drop  > 2.8×ATR14   — sustained flash-crash
"""
import logging
import math

from .strategies.base import BaseStrategy

logger = logging.getLogger("health_monitor")

_INTERVAL_MS  = 3 * 60 * 1000    # 3 minutes (180 s)
_DEFAULT_SCORE = 60               # returned when not enough data (neutral-safe)

# Spike detection thresholds
_SPIKE_TR_MULT      = 2.5   # TR > N×ATR14 on a single bar → v_spike
_SPIKE_DROP_2B      = 2.0   # cumulative drop > N×ATR14 over 2 bars → cascade
_SPIKE_DROP_3B      = 2.8   # cumulative drop > N×ATR14 over 3 bars → cascade
_SPIKE_COOLDOWN_MS  = 30 * 60 * 1000   # 30-min cooldown after trigger
_SPIKE_GUARD_MS     = 2 * 15 * 60 * 1000  # 30-min guard after v_spike (2 × 15m bars)


class HealthMonitor:

    def __init__(self, weak_bars_confirm: int = 3):
        self._weak_bars_confirm = weak_bars_confirm
        self._weak_count  = 0
        self._last_ts_ms  = 0
        self._prev_action = "hold"
        self._spike_last_ms       = 0   # cooldown tracker for spike detector
        self._spike_guard_until_ms = 0  # active guard window after v_spike

        self.last_score  = _DEFAULT_SCORE
        self.last_label  = "neutral"
        self.last_action = "hold"

    # ── Public ───────────────────────────────────────────────────────────────

    @property
    def spike_guard_active(self) -> bool:
        """True for 45 min after a V-spike — strategies hold through the guard."""
        return self._spike_guard_until_ms > 0

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

    # ── Spike / Whipsaw detector (runs every tick) ───────────────────────────

    def check_spike(self, c30m: list, cp: float, now_ms: int) -> dict:
        """
        Runs on every bot tick (60 s).  Detects three danger patterns:

        C1  V-sharp / reversal spike
            TR of last bar OR second-to-last bar > 2.5×ATR14
            Catches single-candle flash spikes and V-reversal bodies.

        C2  Rapid cascade drop
            Cumulative close-to-close fall over 2 bars > 2.0×ATR14
            OR over 3 bars > 2.8×ATR14.

        C3  Whipsaw
            ≥ 4 direction changes in the last 6 bars  AND
            avg per-bar swing > 0.3×ATR14.
            Catches choppy markets that would repeatedly trigger SL.

        Returns:
            {"detected": bool, "type": str, "reason": str}
        Cooldown: once triggered, silent for 30 minutes.
        """
        # Expire spike guard if window has passed
        if self._spike_guard_until_ms > 0 and now_ms > self._spike_guard_until_ms:
            self._spike_guard_until_ms = 0
            logger.info("[Spike] v_spike guard expired — normal exits resumed")

        # Cooldown guard — only enforced after the first trigger (_spike_last_ms > 0)
        if self._spike_last_ms > 0 and (now_ms - self._spike_last_ms) < _SPIKE_COOLDOWN_MS:
            return {"detected": False, "type": "", "reason": "cooldown"}

        n = len(c30m)
        if n < 10:
            return {"detected": False, "type": "", "reason": ""}

        atr14 = BaseStrategy.atr(c30m, 14)
        atr_v = float(atr14[n - 1])
        if math.isnan(atr_v) or atr_v <= 0:
            return {"detected": False, "type": "", "reason": ""}

        # ── C1: V-spike — extreme True Range on last 2 bars ─────────────
        # Single-candle flash spike (wick-only move, then reversal).
        # ACTION: activate spike_guard (hold positions), do NOT close.
        for offset in (1, 2):
            if n <= offset:
                continue
            bar  = c30m[-(offset)]
            prev = c30m[-(offset + 1)]
            tr   = max(
                float(bar.high) - float(bar.low),
                abs(float(bar.high) - float(prev.close)),
                abs(float(bar.low)  - float(prev.close)),
            )
            if tr > _SPIKE_TR_MULT * atr_v:
                reason = (f"V-spike bar[-{offset}] "
                          f"TR={tr:.2f} > {_SPIKE_TR_MULT}×ATR({atr_v:.2f})")
                self._spike_last_ms       = now_ms
                self._spike_guard_until_ms = now_ms + _SPIKE_GUARD_MS  # 45-min guard
                logger.warning("[Spike-C1] %s  guard=30min", reason)
                return {"detected": True, "type": "v_spike", "reason": reason}

        # ── C2: cascade drop — sustained multi-bar fall ──────────────────
        # Real directional move, not a wick.  ACTION: close positions.
        if n >= 3:
            drop_2b = float(c30m[-3].close) - float(c30m[-1].close)
            if drop_2b > _SPIKE_DROP_2B * atr_v:
                reason = (f"Cascade 2-bar drop {drop_2b:.2f} "
                          f"> {_SPIKE_DROP_2B}×ATR({atr_v:.2f})")
                self._spike_last_ms = now_ms
                logger.warning("[Spike-C2] %s", reason)
                return {"detected": True, "type": "cascade", "reason": reason}

        if n >= 4:
            drop_3b = float(c30m[-4].close) - float(c30m[-1].close)
            if drop_3b > _SPIKE_DROP_3B * atr_v:
                reason = (f"Cascade 3-bar drop {drop_3b:.2f} "
                          f"> {_SPIKE_DROP_3B}×ATR({atr_v:.2f})")
                self._spike_last_ms = now_ms
                logger.warning("[Spike-C2b] %s", reason)
                return {"detected": True, "type": "cascade", "reason": reason}

        # ── C3: whipsaw — ≥ 4 direction changes in last 6 bars ──────────
        # Choppy indecision.  ACTION: block new entries, hold existing.
        if n >= 7:
            recent = [float(c.close) for c in c30m[-7:]]
            dir_changes = 0
            swings      = []
            for i in range(1, len(recent) - 1):
                d_prev = recent[i]     - recent[i - 1]
                d_next = recent[i + 1] - recent[i]
                if d_prev * d_next < 0:
                    dir_changes += 1
                swings.append(abs(recent[i] - recent[i - 1]))
            avg_swing = sum(swings) / len(swings) if swings else 0.0

            if dir_changes >= 4 and avg_swing > 0.3 * atr_v:
                reason = (f"Whipsaw: {dir_changes} reversals/6 bars, "
                          f"avg_swing={avg_swing:.2f} (0.3×ATR={0.3*atr_v:.2f})")
                self._spike_last_ms = now_ms
                logger.warning("[Spike-C3] %s", reason)
                return {"detected": True, "type": "whipsaw", "reason": reason}

        return {"detected": False, "type": "", "reason": ""}

    # ── Description helpers ───────────────────────────────────────────────────

    @staticmethod
    def label_emoji(label: str) -> str:
        return {"bull": "🟢", "neutral": "🟡", "weak": "🟠", "strong_weak": "🔴"}.get(label, "⚪")

    def summary(self) -> str:
        emoji = self.label_emoji(self.last_label)
        return (f"{emoji} Health {self.last_score}/100 "
                f"({self.last_label}) → {self.last_action}")
