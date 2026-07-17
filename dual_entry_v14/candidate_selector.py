"""Candidate Selection Engine (spec §21)."""
from __future__ import annotations

from typing import Optional

from .enums import ReasonCode, SetupType
from .models import SignalCandidate, SymbolState


class CandidateSelector:
    def select(self, pullback_candidate: Optional[SignalCandidate],
               momentum_candidate: Optional[SignalCandidate],
               state: SymbolState) -> Optional[SignalCandidate]:
        if state.has_open_position or state.has_pending_order:
            return None
        pb, mo = pullback_candidate, momentum_candidate
        if pb is None and mo is None:
            return None
        # opposite directions passing together -> directional ambiguity
        if pb is not None and mo is not None and pb.direction != mo.direction:
            pb.reason_codes.append(ReasonCode.REJECT_DIRECTIONAL_AMBIGUITY.value)
            mo.reason_codes.append(ReasonCode.REJECT_DIRECTIONAL_AMBIGUITY.value)
            return None
        if pb is None:
            return mo
        if mo is None:
            return pb
        # same direction, both passed
        if pb.ready_for_execution and not mo.ready_for_execution:
            return pb
        if mo.ready_for_execution and not pb.ready_for_execution:
            return mo
        major_break = any(r.startswith("MAJOR:True") for r in mo.reason_codes)
        if major_break and not pb.ready_for_execution:
            return mo
        if abs(pb.edge_score - mo.edge_score) <= 2.0:
            return pb                     # prefer pullback when close
        return pb if pb.edge_score > mo.edge_score else mo
