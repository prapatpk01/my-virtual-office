"""
Layer 7: Expectancy Engine — Trade Quality Gate (part 2 of 2).

Even a high-Confidence setup gets skipped here if this (regime, strategy)
combination has a historically negative edge. Confidence answers "does
this setup look right"; Expectancy answers "has this kind of setup
actually made money" — both must pass.

Uses the Adaptive Learning Engine's trade journal, filtered to the
current (regime, strategy) composite key, over a rolling window.
With too little history to judge (< min_trades), it passes through
neutrally rather than blocking trading indefinitely while the system
is still learning.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExpectancyResult:
    tradeable: bool
    expectancy_r: float
    win_rate: float
    profit_factor: Optional[float]
    avg_win_r: float
    avg_loss_r: float
    kelly_fraction: float
    monte_carlo_positive_pct: float
    sample_size: int
    reason: str = ""
    detail: dict = field(default_factory=dict)


class ExpectancyEngine:
    def __init__(
        self,
        min_trades: int = 20,
        lookback: int = 100,
        min_expectancy_r: float = 0.02,
        min_kelly: float = -0.05,
        mc_samples: int = 200,
    ):
        self.min_trades = min_trades
        self.lookback = lookback
        self.min_expectancy_r = min_expectancy_r
        self.min_kelly = min_kelly
        self.mc_samples = mc_samples

    def evaluate(self, journal: list, regime_key: str) -> ExpectancyResult:
        matching = [t for t in journal if getattr(t, "regime", "") == regime_key]
        matching = matching[-self.lookback:]
        n = len(matching)

        if n < self.min_trades:
            return ExpectancyResult(
                tradeable=True, expectancy_r=0.0, win_rate=0.0, profit_factor=None,
                avg_win_r=0.0, avg_loss_r=0.0, kelly_fraction=0.0,
                monte_carlo_positive_pct=100.0, sample_size=n,
                reason=f"Insufficient history ({n}/{self.min_trades}) — passing through neutrally",
            )

        r_list = [t.pnl_r for t in matching]
        wins = [r for r in r_list if r > 0]
        losses = [r for r in r_list if r <= 0]

        win_rate = len(wins) / n
        avg_win_r = sum(wins) / len(wins) if wins else 0.0
        avg_loss_r = abs(sum(losses) / len(losses)) if losses else 0.0

        expectancy_r = win_rate * avg_win_r - (1 - win_rate) * avg_loss_r

        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else None

        # Kelly fraction: f* = W - (1-W)/R, R = avg_win/avg_loss
        rr_ratio = (avg_win_r / avg_loss_r) if avg_loss_r > 0 else avg_win_r
        kelly = win_rate - (1 - win_rate) / rr_ratio if rr_ratio > 0 else -1.0

        # Monte Carlo stability: bootstrap-resample the R sequence and check
        # what fraction of resamples still land with positive total expectancy.
        mc_positive = 0
        if n >= 5:
            rng = random.Random(42)  # deterministic across calls for the same journal
            for _ in range(self.mc_samples):
                sample = [rng.choice(r_list) for _ in range(n)]
                if sum(sample) > 0:
                    mc_positive += 1
            mc_pct = mc_positive / self.mc_samples * 100.0
        else:
            mc_pct = 50.0

        tradeable = (
            expectancy_r >= self.min_expectancy_r
            and kelly >= self.min_kelly
            and mc_pct >= 45.0
        )
        reason = "" if tradeable else (
            f"Negative/weak edge: expectancy={expectancy_r:+.3f}R kelly={kelly:+.3f} "
            f"mc_stability={mc_pct:.0f}%"
        )

        return ExpectancyResult(
            tradeable=tradeable, expectancy_r=round(expectancy_r, 4),
            win_rate=round(win_rate * 100, 1),
            profit_factor=round(profit_factor, 2) if profit_factor is not None else None,
            avg_win_r=round(avg_win_r, 3), avg_loss_r=round(avg_loss_r, 3),
            kelly_fraction=round(kelly, 3), monte_carlo_positive_pct=round(mc_pct, 1),
            sample_size=n, reason=reason,
            detail={"gross_win": round(gross_win, 2), "gross_loss": round(gross_loss, 2)},
        )
