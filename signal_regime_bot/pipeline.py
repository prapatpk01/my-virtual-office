"""Shared 4H Regime -> 1H/15M Bias -> 15M/5M Expert Entry pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import Config
from regime_engine import RegimeEngine, RegimeResult
from bias_engine import BiasEngine, BiasResult, LONG as B_LONG, SHORT as B_SHORT, BOTH as B_BOTH
from entry_engine import EntryEngine, EntryResult, LONG, SHORT, BOTH, NONE


@dataclass
class PipelineResult:
    direction: str
    price: float
    entry_score: float
    blocked_layer: Optional[str]
    reason: str
    regime: RegimeResult
    bias: Optional[BiasResult] = None
    entry: Optional[EntryResult] = None
    size_multiplier: float = 1.0
    round_id: object = None
    round_age_bars: object = None
    used_booster: bool = False
    context: object = None
    booster: object = None


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.regime_engine = RegimeEngine(cfg)
        self.bias_engine = BiasEngine(cfg)
        self.entry_engine = EntryEngine(cfg)

    def evaluate(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame] = None,
        df_30m: Optional[pd.DataFrame] = None,
        symbol: str = "",
    ) -> PipelineResult:
        has_15m = df_15m is not None and len(df_15m) > 0
        has_5m = df_5m is not None and len(df_5m) > 0
        default_price = (
            float(df_5m["close"].iloc[-1]) if has_5m
            else float(df_15m["close"].iloc[-1]) if has_15m
            else 0.0
        )

        regime = self.regime_engine.analyze(df_4h, df_1h)
        base = dict(regime=regime, size_multiplier=regime.size_multiplier)
        c = self.cfg

        if (
            c.commodity_weekend_block_enabled
            and (has_5m or has_15m)
            and any(k in symbol.upper() for k in c.commodity_symbol_keywords)
        ):
            entry_frame = df_5m if has_5m else df_15m
            entry_tf = c.tf_micro if has_5m else c.tf_fast
            bar_close = entry_frame.index[-1] + pd.Timedelta(entry_tf)
            weekday, hour = bar_close.weekday(), bar_close.hour
            halted = (
                (weekday == 4 and hour >= c.commodity_halt_hour_utc)
                or weekday == 5
                or (weekday == 6 and hour < c.commodity_resume_hour_utc)
            )
            if halted:
                return PipelineResult(
                    NONE, default_price, 0.0, "MARKET",
                    f"commodity weekend halt; entries resume Sunday {c.commodity_resume_hour_utc:02d}:00 UTC",
                    **base,
                )

        bias = self.bias_engine.analyze(df_1h, df_15m, df_5m, regime.label)
        if bias.direction == B_LONG:
            side = LONG
        elif bias.direction == B_SHORT:
            side = SHORT
        elif bias.direction == B_BOTH:
            side = BOTH
        else:
            return PipelineResult(
                NONE, default_price, 0.0, "BIAS", bias.reason,
                bias=bias, **base,
            )

        entry = self.entry_engine.analyze(
            df_30m, df_15m, df_5m, side, symbol,
            df_1h=df_1h, df_4h=df_4h, regime=regime, bias=bias,
        )
        price = entry.price if entry.price > 0 else default_price
        if entry.allow_entry and entry.direction in (LONG, SHORT):
            return PipelineResult(
                entry.direction, price, entry.entry_score, None, entry.reason,
                bias=bias, entry=entry, **base,
            )
        return PipelineResult(
            NONE, price, entry.entry_score, "ENTRY", entry.reason,
            bias=bias, entry=entry, **base,
        )


SignalEngine = Pipeline
