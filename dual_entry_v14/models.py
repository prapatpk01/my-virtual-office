"""Core dataclasses for DUAL ENTRY PRECISION V1.4.

Every model is a plain dataclass (JSON-serializable via asdict) so the
state store can persist atomically and the backtest can replay them.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from .enums import (
    Direction, PatternStatus, PatternType, PullbackType, ReasonCode,
    SetupType, SymbolStatus, ZoneType,
)


# ── Market data ──────────────────────────────────────────────────────────────

@dataclass
class Candle:
    timestamp: int          # bar OPEN time, epoch ms, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return max(self.high - self.low, 1e-12)

    @property
    def is_bull(self) -> bool:
        return self.close > self.open

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def bull_close_quality(self) -> float:
        return (self.close - self.low) / self.range

    @property
    def bear_close_quality(self) -> float:
        return (self.high - self.close) / self.range


# ── Swings / structure ───────────────────────────────────────────────────────

@dataclass
class SwingPoint:
    timeframe: str
    swing_type: str          # "high" | "low"
    price: float
    timestamp: int           # bar-open ms of the pivot bar
    confirmed_at: int        # bar-open ms of the bar that CONFIRMED it (pivot + right bars)

    strength: float = 0.0
    confirmed: bool = True
    broken: bool = False

    reaction_bars: int = 0
    displacement_atr: float = 0.0
    retest_count: int = 0


@dataclass
class StructureEvent:
    timeframe: str
    event_type: str          # "BOS" | "CHOCH" | "FALSE_BOS"
    direction: str           # "LONG" | "SHORT" (the direction the event supports)
    level: float
    confirmed_at: int        # bar-open ms of the confirming close

    body_atr: float = 0.0
    volume_ratio: float = 1.0
    displacement_quality: float = 0.0

    held_after_break: bool = True
    invalidated: bool = False


@dataclass
class StructureView:
    """One timeframe's structure snapshot."""
    timeframe: str
    state: str                                   # StructureState value
    swings: list = field(default_factory=list)   # list[SwingPoint]
    events: list = field(default_factory=list)   # list[StructureEvent] (newest last)
    last_bos: Optional[StructureEvent] = None
    last_choch: Optional[StructureEvent] = None
    last_false_bos: Optional[StructureEvent] = None
    quality: float = 0.0                         # 0-100 structure quality

    def recent_choch_against(self, direction: str, within_bars_ms: int,
                             now_ms: int) -> bool:
        """Confirmed opposite CHOCH still active (not superseded)."""
        if self.last_choch is None:
            return False
        if self.last_choch.direction == direction:
            return False
        return (now_ms - self.last_choch.confirmed_at) <= within_bars_ms


# ── Zones ────────────────────────────────────────────────────────────────────

@dataclass
class PriceZone:
    zone_id: str
    timeframe: str
    zone_type: str           # ZoneType value

    upper_price: float
    lower_price: float
    center_price: float
    width: float

    strength: float = 0.0    # 0-100 zone score
    touches: int = 0
    freshness: float = 15.0

    created_at: int = 0
    last_tested_at: Optional[int] = None

    broken: bool = False
    flipped: bool = False
    invalidated: bool = False

    source_swing_id: Optional[str] = None
    source_structure_event: Optional[str] = None

    reaction_strength: float = 0.0
    displacement_strength: float = 0.0

    @property
    def is_resistance_like(self) -> bool:
        return self.zone_type in (
            ZoneType.MAJOR_RESISTANCE.value, ZoneType.MINOR_RESISTANCE.value,
            ZoneType.SUPPLY.value, ZoneType.RANGE_HIGH.value,
            ZoneType.FLIPPED_RESISTANCE.value, ZoneType.PREVIOUS_DAY_HIGH.value,
        )

    @property
    def is_support_like(self) -> bool:
        return self.zone_type in (
            ZoneType.MAJOR_SUPPORT.value, ZoneType.MINOR_SUPPORT.value,
            ZoneType.DEMAND.value, ZoneType.RANGE_LOW.value,
            ZoneType.FLIPPED_SUPPORT.value, ZoneType.BREAKOUT_RETEST.value,
            ZoneType.PREVIOUS_DAY_LOW.value,
        )

    def contains(self, price: float) -> bool:
        return self.lower_price <= price <= self.upper_price


@dataclass
class SupplyDemandZone:
    zone_id: str
    direction: str           # "LONG" (demand) | "SHORT" (supply)

    base_start: int
    base_end: int

    proximal_line: float
    distal_line: float

    departure_strength: float = 0.0
    bos_confirmed: bool = False
    freshness: float = 15.0
    mitigation_count: int = 0


# ── Patterns ────────────────────────────────────────────────────────────────

@dataclass
class PatternContext:
    pattern_id: str
    timeframe: str
    pattern_type: str
    direction: str
    status: str              # PatternStatus value

    start_time: int = 0
    confirmation_time: Optional[int] = None

    boundary_upper: Optional[float] = None
    boundary_lower: Optional[float] = None
    neckline: Optional[float] = None

    breakout_level: Optional[float] = None
    invalidation_level: Optional[float] = None

    quality_score: float = 0.0
    compression_score: float = 0.0
    volume_score: float = 0.0
    structure_score: float = 0.0
    zone_confluence_score: float = 0.0


# ── Candidates / plans ──────────────────────────────────────────────────────

@dataclass
class SignalCandidate:
    symbol: str
    direction: str
    setup_type: str

    score: float
    threshold: float
    edge_score: float

    entry_reference: float
    structure_stop: float
    target_reference: float

    breakout_level: Optional[float]
    retest_level: Optional[float]
    invalidation_level: float

    signal_timestamp: int          # entry-TF closed-bar OPEN ms
    signal_expiry: int             # epoch ms

    htf_structure: str
    bias: str
    regime: str

    nearest_support: Optional[float]
    nearest_resistance: Optional[float]
    structure_room_r: float

    active_zone: Optional[PriceZone]
    zone_score: float

    pattern_type: Optional[str]
    pattern_status: Optional[str]

    candle_pattern: Optional[str]
    candle_quality: float
    candle_location_score: float

    risk_modifier: float
    reason_codes: list = field(default_factory=list)

    ready_for_execution: bool = True
    stop_candidates: list = field(default_factory=list)   # [(name, price)] setup-relevant stops

    @property
    def signal_key(self) -> tuple:
        return (self.symbol, "15m", self.signal_timestamp, self.setup_type, self.direction)

    @property
    def client_order_id(self) -> str:
        return hashlib.sha256(str(self.signal_key).encode()).hexdigest()[:24]


@dataclass
class TradePlan:
    is_valid: bool
    reason_codes: list = field(default_factory=list)

    entry_reference: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    quantity: float = 0.0            # base units
    contracts: float = 0.0           # exchange contracts
    risk_cash: float = 0.0
    risk_distance: float = 0.0
    effective_risk_distance: float = 0.0
    planned_rr: float = 0.0
    risk_modifier: float = 1.0


@dataclass
class GateResult:
    valid: bool
    reason_codes: list = field(default_factory=list)
    risk_modifier: float = 1.0
    detail: str = ""


# ── Per-symbol state (restart-safe) ─────────────────────────────────────────

@dataclass
class SymbolState:
    symbol: str
    status: str = SymbolStatus.IDLE.value

    setup_type: str = ""
    setup_direction: str = ""

    setup_started_at: Optional[int] = None
    setup_started_bar: Optional[int] = None
    setup_age_bars: int = 0

    setup_low: Optional[float] = None
    setup_high: Optional[float] = None

    breakout_level: Optional[float] = None
    retest_level: Optional[float] = None
    invalidation_level: Optional[float] = None

    active_zone_id: Optional[str] = None
    active_zone_timeframe: Optional[str] = None
    active_zone_type: Optional[str] = None
    active_zone_upper: Optional[float] = None
    active_zone_lower: Optional[float] = None
    active_zone_score: float = 0.0

    pattern_type: Optional[str] = None
    pattern_status: Optional[str] = None
    pattern_breakout_level: Optional[float] = None
    pattern_invalidation_level: Optional[float] = None

    planned_entry: Optional[float] = None
    planned_stop: Optional[float] = None
    planned_target: Optional[float] = None
    planned_risk_distance: Optional[float] = None
    planned_rr: Optional[float] = None

    actual_entry: Optional[float] = None
    actual_quantity: Optional[float] = None
    active_stop: Optional[float] = None
    active_target: Optional[float] = None
    initial_risk: Optional[float] = None
    entry_bar_ts: Optional[int] = None
    entry_fee: float = 0.0
    holding_bars: int = 0
    mfe_r: float = 0.0
    mae_r: float = 0.0

    breakeven_moved: bool = False
    early_exit_sent: bool = False

    pending_order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    position_id: Optional[str] = None

    last_processed_candle: Optional[tuple] = None
    last_exit_bar: Optional[int] = None
    last_signal_key: Optional[str] = None

    previous_regime: Optional[str] = None
    candidate_regime: Optional[str] = None
    candidate_regime_count: int = 0

    consecutive_losses: int = 0
    cooldown_until: Optional[int] = None        # epoch ms
    cooldown_until_bar: Optional[int] = None    # entry-TF bar-open ms

    shock_lockout_until_bar: Optional[int] = None

    state_version: int = 0
    last_reconciled_at: Optional[int] = None

    @property
    def has_open_position(self) -> bool:
        return self.status in (SymbolStatus.LONG_OPEN.value, SymbolStatus.SHORT_OPEN.value)

    @property
    def has_pending_order(self) -> bool:
        return self.status == SymbolStatus.ORDER_PENDING.value or self.pending_order_id is not None

    def cooldown_active(self, now_ms: int, current_bar_ts: Optional[int]) -> bool:
        if self.cooldown_until is not None and now_ms < self.cooldown_until:
            return True
        if (self.cooldown_until_bar is not None and current_bar_ts is not None
                and current_bar_ts <= self.cooldown_until_bar):
            return True
        return False

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("last_processed_candle") is not None:
            d["last_processed_candle"] = list(d["last_processed_candle"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SymbolState":
        d = dict(d)
        if d.get("last_processed_candle") is not None:
            d["last_processed_candle"] = tuple(d["last_processed_candle"])
        known = {f for f in cls.__dataclass_fields__}   # tolerate old/new fields
        return cls(**{k: v for k, v in d.items() if k in known})


# ── Trade record ─────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    setup_type: str
    direction: str

    signal_time: int
    entry_time: int
    exit_time: int

    signal_score: float
    threshold: float
    edge_score: float

    entry_price: float
    stop_price: float
    target_price: float
    exit_price: float

    initial_risk: float
    actual_rr: float

    pnl_cash: float
    pnl_percent: float
    result_r: float

    exit_reason: str

    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    holding_bars: int = 0

    regime_at_entry: str = ""
    bias_at_entry: str = ""
    macro_structure_at_entry: str = ""

    active_zone_type: Optional[str] = None
    zone_score: float = 0.0

    pattern_type: Optional[str] = None
    candle_pattern: Optional[str] = None

    structure_room_r: float = 0.0
    slippage: float = 0.0
    fees: float = 0.0


def utc_ms(dt: Optional[datetime] = None) -> int:
    import time as _t
    return int(_t.time() * 1000) if dt is None else int(dt.timestamp() * 1000)
