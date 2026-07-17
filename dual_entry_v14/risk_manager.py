"""Risk Manager (spec §23) — deterministic structure stops, effective-risk
sizing (stop + slippage + fees), structure-aware targets, actual-RR check.
"""
from __future__ import annotations

from typing import Optional

from .config import Config
from .enums import ReasonCode, SetupType
from .models import GateResult, SignalCandidate, TradePlan


class RiskManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ── deterministic stop selection (spec 23.1/23.2/23.3) ──────────────────
    def select_stop(self, candidate: SignalCandidate, atr: float) -> tuple:
        """Returns (stop_price, reason_codes). Chooses the NEAREST candidate
        that still truly invalidates the thesis, buffered; widens too-tight
        stops behind the next structure; rejects if nothing structural fits."""
        c = self.cfg
        long = candidate.direction == "LONG"
        entry = candidate.entry_reference
        buf = atr * c.stop_buffer_atr
        codes: list = []

        cands = list(candidate.stop_candidates)
        # already filtered to the correct side + sorted nearest-first by engines
        if not cands:
            return 0.0, [ReasonCode.REJECT_STOP_TOO_WIDE.value + ":no_structure"]

        chosen: Optional[float] = None
        for name, level in cands:
            stop = level - buf if long else level + buf
            dist_atr = abs(entry - stop) / max(atr, 1e-12)
            if dist_atr < c.min_stop_atr:
                continue                      # too tight — try next structure out
            if dist_atr > c.max_stop_atr:
                break                         # everything further is even wider
            chosen = stop
            codes.append(f"STOP:{name}")
            break

        if chosen is None:
            # nothing in band: widen from the tightest candidate if that lands in band
            name, level = cands[0]
            stop = level - buf if long else level + buf
            dist_atr = abs(entry - stop) / max(atr, 1e-12)
            if dist_atr < c.min_stop_atr:
                stop = entry - atr * c.min_stop_atr if long else entry + atr * c.min_stop_atr
                dist_atr = c.min_stop_atr
                codes.append("STOP:WIDENED_TO_MIN")
            if dist_atr > c.max_stop_atr:
                hq = candidate.zone_score >= c.zone_hq_score
                if hq and dist_atr <= c.max_stop_atr * 1.15:
                    codes.append("STOP:WIDE_HQ_REDUCED_RISK")
                    return stop, codes
                return 0.0, [ReasonCode.REJECT_STOP_TOO_WIDE.value]
            chosen = stop
        return chosen, codes

    # ── plan (spec 23.4/23.5) ────────────────────────────────────────────────
    def build_trade_plan(self, candidate: SignalCandidate, account: dict,
                         market: dict, portfolio_modifier: float = 1.0,
                         module_modifier: float = 1.0) -> TradePlan:
        c = self.cfg
        long = candidate.direction == "LONG"
        atr = float(market.get("atr", 0.0)) or abs(candidate.entry_reference) * 0.005
        entry = candidate.entry_reference

        stop, stop_codes = self.select_stop(candidate, atr)
        if stop <= 0:
            return TradePlan(False, stop_codes)
        reduced = 0.6 if "STOP:WIDE_HQ_REDUCED_RISK" in stop_codes else 1.0

        stop_distance = abs(entry - stop)
        slip = atr * c.expected_slippage_atr
        fee_dist = entry * c.fee_rate * 2.0            # round-trip fee as distance
        effective_risk_distance = stop_distance + 2 * slip + fee_dist

        equity = float(account.get("equity", 0.0))
        risk_mod = candidate.risk_modifier * portfolio_modifier * module_modifier * reduced
        risk_cash = equity * c.risk_per_trade * risk_mod
        if risk_cash <= 0:
            return TradePlan(False, [f"{ReasonCode.REJECT_TOTAL_RISK.value}:zero_risk_cash"])

        qty = risk_cash / max(effective_risk_distance, 1e-12)

        # market rules
        lot = float(market.get("lot_step", 0.0)) or 0.0
        ct = float(market.get("contract_size", 1.0)) or 1.0
        min_qty = float(market.get("min_qty", 0.0)) or 0.0
        tick = float(market.get("tick_size", 0.0)) or 0.0
        min_notional = float(market.get("min_notional", 0.0)) or 0.0

        contracts = qty / ct
        if lot > 0:
            contracts = int(contracts / lot) * lot     # floor to lot step
        if contracts <= 0 or (min_qty and contracts < min_qty):
            return TradePlan(False, [f"{ReasonCode.REJECT_EXECUTION_QUALITY.value}:below_min_qty"])
        qty = contracts * ct
        if min_notional and qty * entry < min_notional:
            return TradePlan(False, [f"{ReasonCode.REJECT_EXECUTION_QUALITY.value}:below_min_notional"])

        # margin check
        lev = max(1, int(account.get("leverage", c.leverage)))
        margin_needed = qty * entry / lev
        free = float(account.get("free_margin", equity))
        if margin_needed > free * 0.95:
            scale = (free * 0.95) / max(margin_needed, 1e-12)
            contracts = int(contracts * scale / max(lot, 1e-12)) * lot if lot > 0 else contracts * scale
            if contracts <= 0 or (min_qty and contracts < min_qty):
                return TradePlan(False, [f"{ReasonCode.REJECT_TOTAL_RISK.value}:margin"])
            qty = contracts * ct

        # target: min(base RR target, structure target) — spec 23.5. The base
        # target is built from the EFFECTIVE risk distance (stop + slippage +
        # round-trip fees) so the planned RR is net of costs — otherwise every
        # tight-stop trade fails the min-RR check purely on fee drag.
        base_target = entry + effective_risk_distance * c.risk_reward if long \
            else entry - effective_risk_distance * c.risk_reward
        target = base_target
        struct_ref = candidate.nearest_resistance if long else candidate.nearest_support
        if struct_ref is not None:
            struct_target = struct_ref - atr * c.target_buffer_atr if long \
                else struct_ref + atr * c.target_buffer_atr
            target = min(base_target, struct_target) if long else max(base_target, struct_target)

        if tick > 0:
            stop = round(stop / tick) * tick
            target = round(target / tick) * tick

        rr = abs(target - entry) / max(effective_risk_distance, 1e-12)
        if rr < c.min_acceptable_rr:
            return TradePlan(False, [ReasonCode.REJECT_LOW_RR.value + f":{rr:.2f}"])

        return TradePlan(
            True, stop_codes, entry_reference=entry, stop_price=stop, target_price=target,
            quantity=qty, contracts=contracts, risk_cash=risk_cash,
            risk_distance=stop_distance, effective_risk_distance=effective_risk_distance,
            planned_rr=rr, risk_modifier=risk_mod,
        )

    # ── post-fill re-validation (spec 23.5 end) ──────────────────────────────
    def revalidate_after_fill(self, plan: TradePlan, fill_price: float,
                              direction: str, atr: float) -> GateResult:
        c = self.cfg
        long = direction == "LONG"
        stop_dist = abs(fill_price - plan.stop_price)
        eff = stop_dist + 2 * atr * c.expected_slippage_atr + fill_price * c.fee_rate * 2
        rr = abs(plan.target_price - fill_price) / max(eff, 1e-12)
        if rr < c.min_acceptable_rr * 0.92:      # small tolerance for fill noise
            return GateResult(False, [ReasonCode.REJECT_LOW_RR.value + f":post_fill_{rr:.2f}"])
        if (long and fill_price <= plan.stop_price) or (not long and fill_price >= plan.stop_price):
            return GateResult(False, [ReasonCode.REJECT_EXECUTION_QUALITY.value + ":fill_through_stop"])
        return GateResult(True, [], detail=f"rr={rr:.2f}")
