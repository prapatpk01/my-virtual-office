"""Portfolio Risk Gate (spec §22)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .enums import ReasonCode
from .models import GateResult, SignalCandidate

RISK_CLUSTERS = {
    "CRYPTO_MAJOR": ("BTC", "ETH"),
    "CRYPTO_HIGH_BETA": ("SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "HYPE"),
    "METALS": ("XAU", "XAG"),
    "ENERGY": ("CL", "NG"),
}


def cluster_of(symbol: str) -> str:
    s = symbol.upper()
    for cluster, keys in RISK_CLUSTERS.items():
        if any(k in s for k in keys):
            return cluster
    return "OTHER"


@dataclass
class OpenPositionInfo:
    symbol: str
    direction: str
    risk_cash: float


class PortfolioRiskManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def evaluate(self, candidate: SignalCandidate, account_state: dict,
                 open_positions: list) -> GateResult:
        c = self.cfg
        codes: list = []
        equity = float(account_state.get("equity", 0.0))
        if equity <= 0:
            return GateResult(False, [f"{ReasonCode.REJECT_DATA_QUALITY.value}:no_equity"])

        if len(open_positions) >= c.max_positions:
            return GateResult(False, [ReasonCode.REJECT_MAX_POSITIONS.value])
        if any(p.symbol == candidate.symbol for p in open_positions):
            return GateResult(False, [ReasonCode.REJECT_OPEN_POSITION.value])

        open_risk = sum(p.risk_cash for p in open_positions)
        new_risk = equity * c.risk_per_trade * candidate.risk_modifier
        if (open_risk + new_risk) / equity > c.max_total_open_risk:
            return GateResult(False, [ReasonCode.REJECT_TOTAL_RISK.value])

        # correlation: same cluster + same direction -> reduce risk (not reject)
        mod = 1.0
        cl = cluster_of(candidate.symbol)
        correlated = [p for p in open_positions
                      if cluster_of(p.symbol) == cl and p.direction == candidate.direction]
        if correlated:
            mod = c.correlated_risk_factor
            # reject only if even the reduced risk breaches the total cap
            if (open_risk + new_risk * mod) / equity > c.max_total_open_risk:
                return GateResult(False, [ReasonCode.REJECT_CORRELATION.value])
            codes.append(f"CORRELATED_RISK_REDUCED:{cl}")

        dd = account_state.get("daily_drawdown_pct")
        if dd is not None and dd <= -0.05:
            return GateResult(False, [f"{ReasonCode.REJECT_TOTAL_RISK.value}:daily_dd"])

        return GateResult(True, codes, risk_modifier=mod)
