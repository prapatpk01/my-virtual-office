"""
Layer 9 + Module 5: Adaptive Learning Engine

Records every trade with full context, then learns:
  - Win rate by regime / session / score band / symbol
  - Which expert scores predict wins vs losses
  - Auto-adjusts DecisionEngine weights per regime
  - Generates performance summary after N trades

Trade journal schema:
  timestamp, symbol, direction, regime, session,
  expert_scores, decision_score, entry_price, exit_price,
  sl, tp, pnl_r, result (win/loss/be), duration_minutes
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .market_intelligence import MarketRegime


@dataclass
class TradeJournalEntry:
    id:              str
    timestamp:       float
    symbol:          str
    direction:       str
    regime:          str
    session:         str
    entry_price:     float
    exit_price:      float
    stop_loss:       float
    take_profit:     float
    expert_scores:   dict   # {trend, momentum, volatility, liquidity, volume, session, correlation}
    decision_score:  float
    confidence_level:str
    pnl_r:           float   # PnL in R multiples
    result:          str     # "win" | "loss" | "be"
    duration_min:    float
    exit_reason:     str
    mtf_bias:        dict    = field(default_factory=dict)
    checklist_pct:   float   = 0.0


class AdaptiveLearningEngine:
    """Learns from trade history and provides adaptive weight recommendations."""

    MIN_TRADES_TO_ADAPT = 30

    def __init__(self, journal_path: str = "trade_journal.json"):
        self.journal_path = journal_path
        self._journal: List[TradeJournalEntry] = []
        self._load_journal()

    # ── Journal I/O ─────────────────────────────────────────────────────

    def record_trade(
        self,
        symbol:           str,
        direction:        str,
        regime:           str,
        session:          str,
        entry_price:      float,
        exit_price:       float,
        stop_loss:        float,
        take_profit:      float,
        expert_scores:    dict,
        decision_score:   float,
        confidence_level: str,
        exit_reason:      str,
        duration_min:     float = 0.0,
        mtf_bias:         dict  = None,
        checklist_pct:    float = 0.0,
    ) -> TradeJournalEntry:
        entry_id = f"{symbol}-{int(time.time())}"

        # Calculate PnL in R
        risk = abs(entry_price - stop_loss)
        if risk > 0:
            if direction == "long":
                pnl_r = (exit_price - entry_price) / risk
            else:
                pnl_r = (entry_price - exit_price) / risk
        else:
            pnl_r = 0.0

        result = "win" if pnl_r >= 0.5 else "loss" if pnl_r < -0.1 else "be"

        entry = TradeJournalEntry(
            id=entry_id,
            timestamp=time.time(),
            symbol=symbol,
            direction=direction,
            regime=regime,
            session=session,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            expert_scores=expert_scores,
            decision_score=decision_score,
            confidence_level=confidence_level,
            pnl_r=round(pnl_r, 3),
            result=result,
            duration_min=round(duration_min, 1),
            exit_reason=exit_reason,
            mtf_bias=mtf_bias or {},
            checklist_pct=checklist_pct,
        )
        self._journal.append(entry)
        self._save_journal()
        return entry

    # ── Analysis ────────────────────────────────────────────────────────

    def win_rate_by_regime(self) -> dict:
        """Win rate per market regime."""
        buckets: dict = defaultdict(lambda: {"wins": 0, "total": 0, "avg_r": 0.0, "r_list": []})
        for t in self._journal:
            b = buckets[t.regime]
            b["total"] += 1
            if t.result == "win":
                b["wins"] += 1
            b["r_list"].append(t.pnl_r)
        return {
            regime: {
                "win_rate": round(v["wins"] / v["total"] * 100, 1) if v["total"] else 0,
                "trades":   v["total"],
                "avg_r":    round(sum(v["r_list"]) / len(v["r_list"]), 2) if v["r_list"] else 0,
                "expectancy": round(sum(v["r_list"]) / len(v["r_list"]), 3) if v["r_list"] else 0,
            }
            for regime, v in buckets.items()
        }

    def win_rate_by_session(self) -> dict:
        buckets: dict = defaultdict(lambda: {"wins": 0, "total": 0, "r_list": []})
        for t in self._journal:
            b = buckets[t.session]
            b["total"] += 1
            if t.result == "win":
                b["wins"] += 1
            b["r_list"].append(t.pnl_r)
        return {
            s: {
                "win_rate": round(v["wins"]/v["total"]*100, 1) if v["total"] else 0,
                "trades":   v["total"],
                "avg_r":    round(sum(v["r_list"])/len(v["r_list"]), 2) if v["r_list"] else 0,
            }
            for s, v in buckets.items()
        }

    def win_rate_by_confidence(self) -> dict:
        buckets: dict = defaultdict(lambda: {"wins": 0, "total": 0, "r_list": []})
        for t in self._journal:
            b = buckets[t.confidence_level]
            b["total"] += 1
            if t.result == "win":
                b["wins"] += 1
            b["r_list"].append(t.pnl_r)
        return {
            c: {
                "win_rate": round(v["wins"]/v["total"]*100, 1) if v["total"] else 0,
                "trades":   v["total"],
                "avg_r":    round(sum(v["r_list"])/len(v["r_list"]), 2) if v["r_list"] else 0,
            }
            for c, v in buckets.items()
        }

    def expert_weight_recommendations(self) -> dict[str, dict]:
        """Per-regime recommendation: which expert scores most strongly predict wins."""
        if len(self._journal) < self.MIN_TRADES_TO_ADAPT:
            return {}

        expert_keys = ["trend", "momentum", "volatility", "liquidity", "volume", "session"]
        recommendations: dict = {}

        # Group by regime
        by_regime: dict = defaultdict(list)
        for t in self._journal:
            by_regime[t.regime].append(t)

        for regime, trades in by_regime.items():
            if len(trades) < 10:
                continue
            contributions: dict[str, float] = {k: 0.0 for k in expert_keys}
            for t in trades:
                is_win = 1.0 if t.result == "win" else -0.5
                for k in expert_keys:
                    score = t.expert_scores.get(k, 50.0)
                    # High score + win → positive; high score + loss → negative
                    contributions[k] += ((score - 50) / 50) * is_win

            # Normalize to 0-1 weights
            total = sum(abs(v) for v in contributions.values()) + 1e-9
            weights = {k: max(0.05, abs(v) / total) for k, v in contributions.items()}
            weight_sum = sum(weights.values())
            weights = {k: round(v / weight_sum, 4) for k, v in weights.items()}

            win_rate = sum(1 for t in trades if t.result == "win") / len(trades) * 100
            recommendations[regime] = {
                "weights": weights,
                "trades":  len(trades),
                "win_rate": round(win_rate, 1),
            }

        return recommendations

    def performance_summary(self) -> dict:
        if not self._journal:
            return {"trades": 0}
        wins    = [t for t in self._journal if t.result == "win"]
        losses  = [t for t in self._journal if t.result == "loss"]
        be      = [t for t in self._journal if t.result == "be"]
        all_r   = [t.pnl_r for t in self._journal]
        win_r   = [t.pnl_r for t in wins]
        loss_r  = [t.pnl_r for t in losses]
        avg_win = sum(win_r)  / len(win_r)  if win_r  else 0
        avg_loss= sum(loss_r) / len(loss_r) if loss_r else 0
        return {
            "total_trades":  len(self._journal),
            "wins":          len(wins),
            "losses":        len(losses),
            "breakeven":     len(be),
            "win_rate_pct":  round(len(wins) / len(self._journal) * 100, 1),
            "expectancy_r":  round(sum(all_r) / len(all_r), 3),
            "avg_win_r":     round(avg_win, 2),
            "avg_loss_r":    round(avg_loss, 2),
            "profit_factor": round(abs(avg_win / avg_loss), 2) if avg_loss else 999,
            "total_r":       round(sum(all_r), 2),
            "by_regime":     self.win_rate_by_regime(),
            "by_session":    self.win_rate_by_session(),
            "by_confidence": self.win_rate_by_confidence(),
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def _load_journal(self):
        if not os.path.exists(self.journal_path):
            return
        try:
            with open(self.journal_path, "r") as f:
                data = json.load(f)
            self._journal = [TradeJournalEntry(**d) for d in data]
        except Exception:
            self._journal = []

    def _save_journal(self):
        try:
            with open(self.journal_path, "w") as f:
                json.dump([asdict(e) for e in self._journal], f, indent=2)
        except Exception:
            pass
