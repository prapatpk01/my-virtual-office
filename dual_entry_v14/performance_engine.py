"""Performance Engine (spec §28) + Module Performance Gate (spec §29)."""
from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import asdict
from typing import Optional

from .config import Config
from .enums import ModuleStatus, SetupType
from .models import TradeRecord

logger = logging.getLogger("dual_entry.perf")


def _stats(trades: list) -> dict:
    closed = [t for t in trades]
    if not closed:
        return {"trades": 0}
    wins = [t for t in closed if t.result_r > 0]
    losses = [t for t in closed if t.result_r <= 0]
    gross_p = sum(t.pnl_cash for t in closed if t.pnl_cash > 0)
    gross_l = sum(t.pnl_cash for t in closed if t.pnl_cash < 0)
    win_rate = len(wins) / len(closed)
    avg_win_r = sum(t.result_r for t in wins) / len(wins) if wins else 0.0
    avg_loss_r = sum(t.result_r for t in losses) / len(losses) if losses else 0.0
    pf = (gross_p / abs(gross_l)) if gross_l != 0 else float("inf")
    expectancy = win_rate * avg_win_r - (1 - win_rate) * abs(avg_loss_r)
    streak = worst = 0
    for t in closed:
        streak = streak + 1 if t.result_r <= 0 else 0
        worst = max(worst, streak)
    return {"trades": len(closed), "win_rate": round(win_rate, 4),
            "profit_factor": round(pf, 3) if pf != float("inf") else "inf",
            "expectancy_r": round(expectancy, 4),
            "avg_win_r": round(avg_win_r, 3), "avg_loss_r": round(avg_loss_r, 3),
            "max_losing_streak": worst,
            "net_pnl": round(sum(t.pnl_cash for t in closed), 2)}


class PerformanceEngine:
    def __init__(self, cfg: Config, state_dir: str):
        self.cfg = cfg
        self.trades: list = []
        self.module_status = {SetupType.FAST_PULLBACK.value: ModuleStatus.ACTIVE.value,
                              SetupType.MOMENTUM.value: ModuleStatus.ACTIVE.value}
        self.shadow_counters = {SetupType.FAST_PULLBACK.value: deque(maxlen=50),
                                SetupType.MOMENTUM.value: deque(maxlen=50)}
        self._path = os.path.join(state_dir, "trades.jsonl")
        os.makedirs(state_dir, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                for line in f:
                    d = json.loads(line)
                    known = set(TradeRecord.__dataclass_fields__)
                    self.trades.append(TradeRecord(**{k: v for k, v in d.items() if k in known}))
        except Exception as e:
            logger.warning("[PERF] trade log load failed: %s", e)

    def record(self, trade: TradeRecord) -> None:
        self.trades.append(trade)
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(trade)) + "\n")
        self._update_module_gate(trade.setup_type)

    def record_shadow(self, setup_type: str, result_r: float) -> None:
        self.shadow_counters.setdefault(setup_type, deque(maxlen=50)).append(result_r)
        self._maybe_reopen(setup_type)

    # ── stats windows (30/50/100) ────────────────────────────────────────────

    def stats(self, n: Optional[int] = None, setup_type: Optional[str] = None,
              direction: Optional[str] = None) -> dict:
        ts = self.trades
        if setup_type:
            ts = [t for t in ts if t.setup_type == setup_type]
        if direction:
            ts = [t for t in ts if t.direction == direction]
        if n:
            ts = ts[-n:]
        return _stats(ts)

    def full_report(self) -> dict:
        return {
            "last_30": self.stats(30), "last_50": self.stats(50), "last_100": self.stats(100),
            "pullback": self.stats(100, SetupType.FAST_PULLBACK.value),
            "momentum": self.stats(100, SetupType.MOMENTUM.value),
            "long": self.stats(100, direction="LONG"),
            "short": self.stats(100, direction="SHORT"),
            "module_status": dict(self.module_status),
        }

    # ── module performance gate (spec §29) ───────────────────────────────────

    def module_risk_modifier(self, setup_type: str) -> Optional[float]:
        """None = paused (no live orders). 1.0 active, reduced factor otherwise."""
        st = self.module_status.get(setup_type, ModuleStatus.ACTIVE.value)
        if st == ModuleStatus.PAUSED.value:
            return None
        if st == ModuleStatus.REDUCED_RISK.value:
            return self.cfg.module_reduced_risk_factor
        return 1.0

    def _update_module_gate(self, setup_type: str) -> None:
        c = self.cfg
        s50 = self.stats(c.module_min_trades_paused, setup_type)
        s30 = self.stats(c.module_min_trades_reduced, setup_type)
        if s50.get("trades", 0) >= c.module_min_trades_paused:
            pf = s50.get("profit_factor", 0)
            pf = float("inf") if pf == "inf" else float(pf)
            if pf < c.module_pf_reduced_low and s50.get("expectancy_r", 0) < 0:
                if self.module_status[setup_type] != ModuleStatus.PAUSED.value:
                    logger.warning("[MODULE] %s PAUSED (pf=%.2f)", setup_type, pf)
                self.module_status[setup_type] = ModuleStatus.PAUSED.value
                return
        if s30.get("trades", 0) >= c.module_min_trades_reduced:
            pf = s30.get("profit_factor", 0)
            pf = float("inf") if pf == "inf" else float(pf)
            exp = s30.get("expectancy_r", 0)
            if c.module_pf_reduced_low <= pf <= c.module_pf_reduced_high or abs(exp) < 0.02:
                self.module_status[setup_type] = ModuleStatus.REDUCED_RISK.value
                return
        if self.module_status[setup_type] != ModuleStatus.PAUSED.value:
            self.module_status[setup_type] = ModuleStatus.ACTIVE.value

    def _maybe_reopen(self, setup_type: str) -> None:
        c = self.cfg
        if self.module_status.get(setup_type) != ModuleStatus.PAUSED.value:
            return
        shadow = self.shadow_counters.get(setup_type, [])
        if len(shadow) >= c.module_shadow_reopen_signals:
            rs = list(shadow)
            wins = sum(1 for r in rs if r > 0)
            gp = sum(r for r in rs if r > 0)
            gl = abs(sum(r for r in rs if r <= 0)) or 1e-9
            if gp / gl > 1.0 and (wins / len(rs)) * (gp / max(wins, 1)) > 0:
                logger.info("[MODULE] %s re-opened from shadow", setup_type)
                self.module_status[setup_type] = ModuleStatus.REDUCED_RISK.value
                self.shadow_counters[setup_type].clear()

    def manual_reset(self, setup_type: str) -> None:
        self.module_status[setup_type] = ModuleStatus.ACTIVE.value
