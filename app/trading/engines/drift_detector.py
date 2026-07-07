"""
Module 10: Monitoring & Drift Detection

Detects when market behavior has shifted enough that the current model
is degrading in performance (concept drift). Triggers:
  - Warning when rolling win rate drops significantly vs. baseline
  - Alert when expected value (avg R) turns negative over recent window
  - Regime shift detection: dominant regime changed significantly
  - Volatility regime change: ATR percentile shifts

Actions on drift:
  - WARN:   Log warning, flag in dashboard
  - RETRAIN: Recommend weight re-optimization
  - PAUSE:   Suspend trading until re-evaluation
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class DriftAction(str, Enum):
    NONE    = "none"
    WARN    = "warn"
    RETRAIN = "retrain"
    PAUSE   = "pause"


@dataclass
class DriftAlert:
    action:       DriftAction
    severity:     float         # 0-1
    message:      str
    metric:       str
    current_val:  float
    baseline_val: float
    timestamp:    float = field(default_factory=time.time)


class DriftDetector:
    """Monitors model performance for concept drift."""

    def __init__(
        self,
        baseline_win_rate:   float = 0.50,  # Expected win rate at launch
        baseline_avg_r:      float = 0.30,  # Expected avg R per trade
        warn_wr_drop:        float = 0.10,  # Warn if win rate drops 10%
        pause_wr_drop:       float = 0.20,  # Pause if win rate drops 20%
        rolling_window:      int   = 20,    # Evaluate over last N trades
        retrain_window:      int   = 50,    # Retrain suggestion window
    ):
        self.baseline_wr  = baseline_win_rate
        self.baseline_r   = baseline_avg_r
        self.warn_drop    = warn_wr_drop
        self.pause_drop   = pause_wr_drop
        self.roll_window  = rolling_window
        self.retrain_win  = retrain_window
        self._alerts: List[DriftAlert] = []
        self._regime_history: list = []
        self._vol_history:    list = []

    def evaluate(
        self,
        recent_trades: list,           # last N TradeJournalEntry objects
        current_regime: str = "",
        current_atr_pct: float = 0.0,
    ) -> List[DriftAlert]:
        """Run all drift checks and return any new alerts."""
        new_alerts: List[DriftAlert] = []

        if len(recent_trades) < self.roll_window:
            return new_alerts

        window = recent_trades[-self.roll_window:]
        wins   = sum(1 for t in window if t.result == "win")
        roll_wr= wins / len(window)
        roll_r = sum(t.pnl_r for t in window) / len(window)

        # ── Win rate drift ─────────────────────────────────────────────────
        wr_drop = self.baseline_wr - roll_wr
        if wr_drop >= self.pause_drop:
            new_alerts.append(DriftAlert(
                action=DriftAction.PAUSE,
                severity=min(1.0, wr_drop * 3),
                message=f"Win rate dropped {wr_drop*100:.1f}% over last {self.roll_window} trades. PAUSE recommended.",
                metric="win_rate",
                current_val=round(roll_wr, 3),
                baseline_val=self.baseline_wr,
            ))
        elif wr_drop >= self.warn_drop:
            new_alerts.append(DriftAlert(
                action=DriftAction.WARN,
                severity=wr_drop / self.pause_drop,
                message=f"Win rate declined {wr_drop*100:.1f}% (rolling {self.roll_window}). Consider retraining.",
                metric="win_rate",
                current_val=round(roll_wr, 3),
                baseline_val=self.baseline_wr,
            ))

        # ── Negative expectancy ──────────────────────────────────────────────
        if roll_r < 0:
            new_alerts.append(DriftAlert(
                action=DriftAction.RETRAIN,
                severity=min(1.0, abs(roll_r) * 2),
                message=f"Negative expectancy {roll_r:.2f}R over last {self.roll_window} trades. RETRAIN weights.",
                metric="avg_r",
                current_val=round(roll_r, 3),
                baseline_val=self.baseline_r,
            ))

        # ── Regime drift ────────────────────────────────────────────────────
        if current_regime:
            self._regime_history.append(current_regime)
            if len(self._regime_history) > self.retrain_win:
                self._regime_history.pop(0)
            if len(self._regime_history) >= 10:
                recent_regimes = self._regime_history[-10:]
                prior_regimes  = self._regime_history[:10]
                if prior_regimes:
                    dom_recent = max(set(recent_regimes), key=recent_regimes.count)
                    dom_prior  = max(set(prior_regimes),  key=prior_regimes.count)
                    if dom_recent != dom_prior:
                        new_alerts.append(DriftAlert(
                            action=DriftAction.RETRAIN,
                            severity=0.6,
                            message=f"Regime shifted from '{dom_prior}' to '{dom_recent}'. Weights may need update.",
                            metric="regime",
                            current_val=0.0,
                            baseline_val=0.0,
                        ))

        # ── Volatility regime change ──────────────────────────────────────────
        if current_atr_pct > 0:
            self._vol_history.append(current_atr_pct)
            if len(self._vol_history) > self.retrain_win:
                self._vol_history.pop(0)
            if len(self._vol_history) >= 20:
                import numpy as np
                baseline_vol = float(np.median(self._vol_history[:-10]))
                recent_vol   = float(np.median(self._vol_history[-10:]))
                if baseline_vol > 0:
                    vol_change = abs(recent_vol - baseline_vol) / baseline_vol
                    if vol_change > 0.50:
                        new_alerts.append(DriftAlert(
                            action=DriftAction.WARN,
                            severity=min(1.0, vol_change),
                            message=f"Volatility regime changed {vol_change*100:.0f}%. ATR: {baseline_vol:.4f} → {recent_vol:.4f}",
                            metric="volatility",
                            current_val=round(recent_vol, 5),
                            baseline_val=round(baseline_vol, 5),
                        ))

        self._alerts.extend(new_alerts)
        return new_alerts

    def latest_alerts(self, n: int = 10) -> List[DriftAlert]:
        return self._alerts[-n:]

    def highest_severity_action(self) -> DriftAction:
        if not self._alerts:
            return DriftAction.NONE
        recent = self._alerts[-5:]
        if any(a.action == DriftAction.PAUSE for a in recent):
            return DriftAction.PAUSE
        if any(a.action == DriftAction.RETRAIN for a in recent):
            return DriftAction.RETRAIN
        if any(a.action == DriftAction.WARN for a in recent):
            return DriftAction.WARN
        return DriftAction.NONE
