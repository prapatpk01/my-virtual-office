"""
Learning Analysis — deep insight engine over historical signals & trade outcomes.

Turns the raw `fired` / `outcomes` logs kept by `SignalState` into actionable,
data-driven insights: which strategies/symbols/confidence bands actually make
money, whether performance is trending up or down, and plain-language
recommendations. Also used to feed `AISignalStrategy` with a short summary of
"what has worked historically" so its prompt can reason with real track
record data instead of only the current candle window.
"""
from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone


def _win_rate(wins: int, closed: int) -> float | None:
    return round(wins / closed * 100, 1) if closed else None


class LearningAnalysis:
    """
    Computes deep-dive performance analytics from a SignalState's history.

    All methods are pure functions over the `fired`/`outcomes` lists so this
    class has no persistence of its own and can be reused/tested easily.
    """

    def __init__(self, fired: list[dict], outcomes: list[dict]):
        self.fired = fired or []
        self.outcomes = outcomes or []

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def _filter_days(self, days: int) -> tuple[list[dict], list[dict]]:
        cutoff = int(time.time() * 1000) - days * 86_400_000
        fired = [f for f in self.fired if f.get("ts", 0) >= cutoff]
        outcomes = [o for o in self.outcomes if o.get("ts", 0) >= cutoff]
        return fired, outcomes

    def by_symbol(self, outcomes: list[dict]) -> dict:
        data: dict[str, dict] = defaultdict(lambda: {"wins": 0, "losses": 0, "total_r": 0.0})
        for o in outcomes:
            sym = o.get("symbol", "unknown")
            d = data[sym]
            if o["pnl_r"] > 0:
                d["wins"] += 1
            else:
                d["losses"] += 1
            d["total_r"] += o["pnl_r"]
        out = {}
        for sym, d in data.items():
            closed = d["wins"] + d["losses"]
            out[sym] = {
                "trades": closed,
                "wins": d["wins"],
                "losses": d["losses"],
                "win_rate": _win_rate(d["wins"], closed),
                "total_r": round(d["total_r"], 2),
            }
        return out

    def by_confidence(self, outcomes: list[dict]) -> dict:
        """Win-rate bucketed by the confidence recorded on the trade (if present)."""
        data: dict[str, dict] = defaultdict(lambda: {"wins": 0, "losses": 0})
        for o in outcomes:
            c = o.get("confidence")
            if c is None:
                label = "unknown"
            elif c >= 0.8:
                label = "high (≥0.8)"
            elif c >= 0.5:
                label = "medium (0.5-0.8)"
            else:
                label = "low (<0.5)"
            d = data[label]
            if o["pnl_r"] > 0:
                d["wins"] += 1
            else:
                d["losses"] += 1
        out = {}
        for label, d in data.items():
            closed = d["wins"] + d["losses"]
            out[label] = {"trades": closed, "win_rate": _win_rate(d["wins"], closed)}
        return out

    def by_hour(self, outcomes: list[dict]) -> dict:
        """Win-rate bucketed by UTC hour-of-day the trade was closed."""
        data: dict[int, dict] = defaultdict(lambda: {"wins": 0, "losses": 0})
        for o in outcomes:
            ts = o.get("ts")
            if not ts:
                continue
            hour = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).hour
            d = data[hour]
            if o["pnl_r"] > 0:
                d["wins"] += 1
            else:
                d["losses"] += 1
        out = {}
        for hour, d in sorted(data.items()):
            closed = d["wins"] + d["losses"]
            out[hour] = {"trades": closed, "win_rate": _win_rate(d["wins"], closed)}
        return out

    def by_exit_reason(self, outcomes: list[dict]) -> dict:
        data: dict[str, int] = defaultdict(int)
        for o in outcomes:
            data[o.get("reason", "unknown")] += 1
        return dict(data)

    def trend(self, outcomes: list[dict], recent_n: int = 20) -> dict:
        """Compare win-rate of the most recent N trades vs the rest to detect drift."""
        if len(outcomes) < 4:
            return {"direction": "insufficient_data", "recent_win_rate": None, "prior_win_rate": None}
        recent = outcomes[-recent_n:]
        prior = outcomes[:-recent_n] if len(outcomes) > recent_n else []
        recent_wr = _win_rate(sum(1 for o in recent if o["pnl_r"] > 0), len(recent))
        prior_wr = _win_rate(sum(1 for o in prior if o["pnl_r"] > 0), len(prior)) if prior else None
        direction = "flat"
        if prior_wr is not None and recent_wr is not None:
            if recent_wr - prior_wr >= 5:
                direction = "improving"
            elif prior_wr - recent_wr >= 5:
                direction = "declining"
        return {
            "direction": direction,
            "recent_win_rate": recent_wr,
            "prior_win_rate": prior_wr,
            "recent_trades": len(recent),
        }

    # ------------------------------------------------------------------
    # Recommendations (plain-language, derived from the numbers above)
    # ------------------------------------------------------------------

    def recommendations(self, strategy_stats: dict, symbol_stats: dict,
                        confidence_stats: dict, trend_stats: dict) -> list[str]:
        recs: list[str] = []

        # Confidence calibration
        high = confidence_stats.get("high (≥0.8)")
        low = confidence_stats.get("low (<0.5)")
        if high and high["win_rate"] is not None and high["trades"] >= 3:
            if low and low["win_rate"] is not None and high["win_rate"] - low["win_rate"] >= 15:
                recs.append(
                    f"High-confidence signals (≥0.8) win {high['win_rate']}% vs "
                    f"{low['win_rate']}% for low-confidence ones — weight confidence more heavily."
                )
            elif high["win_rate"] < 45:
                recs.append(
                    f"High-confidence signals only win {high['win_rate']}% — confidence scoring "
                    "may be miscalibrated and needs review."
                )

        # Strategy standouts
        ranked = sorted(
            ((s, d) for s, d in strategy_stats.items() if d.get("wins", 0) + d.get("losses", 0) >= 3),
            key=lambda kv: (kv[1].get("win_rate") or 0), reverse=True,
        )
        if ranked:
            best_s, best_d = ranked[0]
            worst_s, worst_d = ranked[-1]
            if best_s != worst_s and (best_d.get("win_rate") or 0) - (worst_d.get("win_rate") or 0) >= 15:
                recs.append(
                    f"'{best_s}' outperforms ({best_d['win_rate']}% WR) vs '{worst_s}' "
                    f"({worst_d['win_rate']}% WR) — consider reducing size on the weaker strategy."
                )

        # Symbol standouts
        sym_ranked = sorted(
            ((s, d) for s, d in symbol_stats.items() if d.get("trades", 0) >= 3),
            key=lambda kv: kv[1].get("total_r", 0), reverse=True,
        )
        if sym_ranked and sym_ranked[0][1]["total_r"] > 0 > sym_ranked[-1][1]["total_r"]:
            recs.append(
                f"'{sym_ranked[0][0]}' is the top contributor (+{sym_ranked[0][1]['total_r']}R) "
                f"while '{sym_ranked[-1][0]}' is dragging performance down "
                f"({sym_ranked[-1][1]['total_r']}R)."
            )

        # Trend
        if trend_stats.get("direction") == "declining":
            recs.append(
                f"Recent win rate ({trend_stats['recent_win_rate']}%) is trending below the "
                f"historical average ({trend_stats['prior_win_rate']}%) — review recent market "
                "conditions or tighten risk."
            )
        elif trend_stats.get("direction") == "improving":
            recs.append(
                f"Recent win rate ({trend_stats['recent_win_rate']}%) is improving over the "
                f"historical average ({trend_stats['prior_win_rate']}%) — current settings are working."
            )

        if not recs:
            recs.append("Not enough closed trades yet for statistically meaningful insights.")
        return recs

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(self, days: int = 30) -> dict:
        fired, outcomes = self._filter_days(days)

        # Per-strategy win-rate (re-derived here so this module is self-contained)
        strat_data: dict[str, dict] = defaultdict(lambda: {"wins": 0, "losses": 0})
        for o in outcomes:
            s = o.get("strategy") or "unknown"
            d = strat_data[s]
            if o["pnl_r"] > 0:
                d["wins"] += 1
            else:
                d["losses"] += 1
        strategy_stats = {}
        for s, d in strat_data.items():
            closed = d["wins"] + d["losses"]
            strategy_stats[s] = {"wins": d["wins"], "losses": d["losses"], "win_rate": _win_rate(d["wins"], closed)}

        symbol_stats = self.by_symbol(outcomes)
        confidence_stats = self.by_confidence(outcomes)
        hour_stats = self.by_hour(outcomes)
        exit_reasons = self.by_exit_reason(outcomes)
        trend_stats = self.trend(outcomes)

        return {
            "period_days": days,
            "total_signals": len(fired),
            "total_closed": len(outcomes),
            "by_strategy": strategy_stats,
            "by_symbol": symbol_stats,
            "by_confidence": confidence_stats,
            "by_hour_utc": hour_stats,
            "by_exit_reason": exit_reasons,
            "trend": trend_stats,
            "recommendations": self.recommendations(
                strategy_stats, symbol_stats, confidence_stats, trend_stats
            ),
        }

    def context_for_ai(self, days: int = 30, max_lines: int = 4) -> str:
        """
        Short plain-text digest suitable for injecting into an LLM prompt so
        strategies like AISignalStrategy can reason using real track record
        rather than only the current candle window.
        """
        insights = self.analyze(days=days)
        if insights["total_closed"] == 0:
            return "No historical trade data yet."
        lines = [
            f"Historical performance (last {days}d, {insights['total_closed']} closed trades):"
        ]
        lines.extend(f"- {r}" for r in insights["recommendations"][:max_lines])
        return "\n".join(lines)
