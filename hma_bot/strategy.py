"""
HMA16 Trend-Follow Bot v3
Timeframe: 15m

Core architecture
-----------------
1) EMA20/EMA50 selects trade direction only.
2) ADX + CHOP + DMI form a soft Trend Quality filter.
3) HMA16 color/state flip triggers entries.
4) HMA16 opposite flip is the primary early exit.
5) Hard TP/SL are both 1.5% from entry as spike protection.
6) Quality filters:
   - EMA separation >= 0.15 ATR
   - HMA16 slope magnitude >= 0.03 ATR
   - Price extension from EMA20 <= 0.80 ATR
   - Trend Quality >= 55/100
7) Closed-candle logic only for indicator-driven entries/exits.

ADX/CHOP/DMI are intentionally permissive. Their job is to reject weak/choppy
HMA flips, not replace the EMA trend gate or HMA16 trigger.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Optional

import numpy as np
import pandas as pd


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class Trend(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"


class ExitReason(str, Enum):
    HMA_FLIP = "HMA_FLIP"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"


@dataclass(frozen=True)
class StrategyConfig:
    timeframe: str = "15m"

    ema_fast_len: int = 20
    ema_slow_len: int = 50
    ema_slope_lookback: int = 1

    hma_len: int = 16
    atr_len: int = 14
    dmi_len: int = 14
    adx_len: int = 14
    chop_len: int = 14

    min_ema_separation_atr: float = 0.15
    min_hma_slope_atr: float = 0.03
    max_chase_atr: float = 0.80

    min_trend_quality: float = 55.0

    # Loose fail-safes only. These are not intended to make the strategy strict.
    adx_hard_floor: float = 10.0
    chop_hard_ceiling: float = 62.0

    take_profit_pct: float = 0.015
    stop_loss_pct: float = 0.015

    require_closed_candle: bool = True


@dataclass
class EntrySignal:
    side: Side
    entry_price: float
    stop_loss: float
    take_profit: float
    trend: Trend

    ema20: float
    ema50: float
    hma16: float
    atr14: float

    adx: float
    plus_di: float
    minus_di: float
    chop: float
    trend_quality: float

    ema_separation_atr: float
    hma_slope_atr: float
    chase_atr: float

    reason: str


@dataclass
class PositionState:
    side: Side
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_bar_time: Optional[pd.Timestamp] = None


@dataclass
class ExitSignal:
    should_exit: bool
    reason: Optional[ExitReason] = None
    exit_price: Optional[float] = None


class HMA16TrendFollowStrategy:
    """
    15m trend-follow strategy.

    Trend direction:
      Bull = EMA20 > EMA50 and EMA20 slope > 0
      Bear = EMA20 < EMA50 and EMA20 slope < 0

    Soft trend quality:
      ADX + CHOP = 0..100 base quality
      DMI alignment = +/-10 adjustment, clamped to 0..100
      Default minimum = 45

    Entry:
      Long = Bull + HMA16 DOWN->UP + quality filters
      Short = Bear + HMA16 UP->DOWN + quality filters

    Exit:
      Primary = opposite HMA16 flip on closed candle
      Safety  = live hard TP/SL at +/-1.5%
    """

    def __init__(self, config: Optional[StrategyConfig] = None) -> None:
        self.cfg = config or StrategyConfig()

    @staticmethod
    def _wma(series: pd.Series, length: int) -> pd.Series:
        if length <= 0:
            raise ValueError("WMA length must be > 0")
        weights = np.arange(1, length + 1, dtype=float)
        weight_sum = float(weights.sum())
        return series.rolling(length).apply(
            lambda values: float(np.dot(values, weights) / weight_sum),
            raw=True,
        )

    @classmethod
    def _hma(cls, series: pd.Series, length: int) -> pd.Series:
        if length < 2:
            raise ValueError("HMA length must be >= 2")
        half = max(1, length // 2)
        root = max(1, int(round(sqrt(length))))
        raw = 2.0 * cls._wma(series, half) - cls._wma(series, length)
        return cls._wma(raw, root)

    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        prev_close = df["close"].shift(1)
        return pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

    @staticmethod
    def _rma(series: pd.Series, length: int) -> pd.Series:
        if length <= 0:
            raise ValueError("RMA length must be > 0")
        return series.ewm(alpha=1.0 / length, adjust=False).mean()

    @classmethod
    def _atr(cls, df: pd.DataFrame, length: int) -> pd.Series:
        return cls._rma(cls._true_range(df), length)

    @classmethod
    def _dmi_adx(
        cls,
        df: pd.DataFrame,
        dmi_len: int,
        adx_len: int,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Wilder DMI/ADX. Returns (+DI, -DI, ADX)."""
        up_move = df["high"].diff()
        down_move = -df["low"].diff()

        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=df.index,
            dtype=float,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=df.index,
            dtype=float,
        )

        tr_rma = cls._rma(cls._true_range(df), dmi_len).replace(0, np.nan)
        plus_di = 100.0 * cls._rma(plus_dm, dmi_len) / tr_rma
        minus_di = 100.0 * cls._rma(minus_dm, dmi_len) / tr_rma

        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        adx = cls._rma(dx.fillna(0.0), adx_len)

        return plus_di, minus_di, adx

    @classmethod
    def _choppiness(cls, df: pd.DataFrame, length: int) -> pd.Series:
        """
        CHOP = 100 * log10(sum(TR,n)/(HH(n)-LL(n))) / log10(n)
        """
        if length <= 1:
            raise ValueError("CHOP length must be > 1")

        tr_sum = cls._true_range(df).rolling(length).sum()
        hh = df["high"].rolling(length).max()
        ll = df["low"].rolling(length).min()
        price_range = (hh - ll).replace(0, np.nan)

        ratio = (tr_sum / price_range).clip(lower=1.0)
        chop = 100.0 * np.log10(ratio) / np.log10(float(length))
        return chop.clip(lower=0.0, upper=100.0)

    def add_indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close"}
        missing = required.difference(candles.columns)
        if missing:
            raise ValueError(f"Missing candle columns: {sorted(missing)}")

        df = candles.copy()

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[list(required)].isna().any().any():
            raise ValueError("OHLC data contains NaN or non-numeric values")

        df["ema20"] = df["close"].ewm(
            span=self.cfg.ema_fast_len,
            adjust=False,
        ).mean()
        df["ema50"] = df["close"].ewm(
            span=self.cfg.ema_slow_len,
            adjust=False,
        ).mean()
        df["hma16"] = self._hma(df["close"], self.cfg.hma_len)
        df["atr14"] = self._atr(df, self.cfg.atr_len)

        plus_di, minus_di, adx = self._dmi_adx(
            df,
            self.cfg.dmi_len,
            self.cfg.adx_len,
        )
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        df["adx"] = adx
        df["adx_rising"] = df["adx"] > df["adx"].shift(1)
        df["chop"] = self._choppiness(df, self.cfg.chop_len)

        lb = self.cfg.ema_slope_lookback
        df["ema20_slope"] = df["ema20"] - df["ema20"].shift(lb)
        df["hma16_slope"] = df["hma16"] - df["hma16"].shift(1)

        # HMA state/color:
        # +1 = blue/up, -1 = orange/down, 0 = flat.
        df["hma_state"] = np.select(
            [df["hma16_slope"] > 0, df["hma16_slope"] < 0],
            [1, -1],
            default=0,
        )

        atr_safe = df["atr14"].replace(0, np.nan)
        df["ema_separation_atr"] = (
            (df["ema20"] - df["ema50"]).abs() / atr_safe
        )
        df["hma_slope_atr"] = df["hma16_slope"].abs() / atr_safe
        df["distance_ema20_atr"] = (
            (df["close"] - df["ema20"]).abs() / atr_safe
        )

        return df

    def classify_trend(self, row: pd.Series) -> Trend:
        if row["ema20"] > row["ema50"] and row["ema20_slope"] > 0:
            return Trend.BULL
        if row["ema20"] < row["ema50"] and row["ema20_slope"] < 0:
            return Trend.BEAR
        return Trend.NEUTRAL

    @staticmethod
    def _flip_up(prev_row: pd.Series, row: pd.Series) -> bool:
        return int(prev_row["hma_state"]) <= 0 and int(row["hma_state"]) > 0

    @staticmethod
    def _flip_down(prev_row: pd.Series, row: pd.Series) -> bool:
        return int(prev_row["hma_state"]) >= 0 and int(row["hma_state"]) < 0

    @staticmethod
    def _adx_score(adx: float) -> float:
        if adx < 12:
            return 0.0
        if adx < 15:
            return 15.0
        if adx < 20:
            return 30.0
        if adx < 25:
            return 40.0
        return 50.0

    @staticmethod
    def _chop_score(chop: float) -> float:
        if chop < 45:
            return 50.0
        if chop < 50:
            return 40.0
        if chop < 55:
            return 30.0
        if chop < 60:
            return 15.0
        return 0.0

    def trend_quality_score(self, row: pd.Series, side: Side) -> float:
        """
        Trend Quality Q = ADX score + CHOP score only, range 0..100.

        DMI (+DI/-DI) is still calculated and reported as confirmation,
        but it does NOT increase or reduce Q.

        Entry requires Q >= min_trend_quality (default 55).
        """
        adx = float(row["adx"])
        chop = float(row["chop"])
        score = self._adx_score(adx) + self._chop_score(chop)
        return float(np.clip(score, 0.0, 100.0))

    def _quality_gate_common(self, row: pd.Series) -> bool:
        values = [
            row["ema_separation_atr"],
            row["hma_slope_atr"],
            row["atr14"],
            row["adx"],
            row["plus_di"],
            row["minus_di"],
            row["chop"],
        ]
        if any(pd.isna(v) for v in values):
            return False

        if row["atr14"] <= 0:
            return False
        if row["ema_separation_atr"] < self.cfg.min_ema_separation_atr:
            return False
        if row["hma_slope_atr"] < self.cfg.min_hma_slope_atr:
            return False

        return True

    def _long_chase_ok(self, row: pd.Series) -> tuple[bool, float]:
        chase = max(
            0.0,
            float((row["close"] - row["ema20"]) / row["atr14"]),
        )
        return chase <= self.cfg.max_chase_atr, chase

    def _short_chase_ok(self, row: pd.Series) -> tuple[bool, float]:
        chase = max(
            0.0,
            float((row["ema20"] - row["close"]) / row["atr14"]),
        )
        return chase <= self.cfg.max_chase_atr, chase

    def generate_entry(
        self,
        candles: pd.DataFrame,
        has_open_position: bool = False,
    ) -> Optional[EntrySignal]:
        """Evaluate the latest CLOSED 15m candle."""
        if has_open_position:
            return None

        min_bars = max(
            self.cfg.ema_slow_len + 5,
            self.cfg.hma_len * 2,
            self.cfg.atr_len + 5,
            self.cfg.dmi_len + self.cfg.adx_len + 5,
            self.cfg.chop_len + 5,
        )
        if len(candles) < min_bars:
            return None

        df = self.add_indicators(candles)
        prev_row = df.iloc[-2]
        row = df.iloc[-1]

        if not self._quality_gate_common(row):
            return None

        trend = self.classify_trend(row)
        entry = float(row["close"])

        if trend == Trend.BULL and self._flip_up(prev_row, row):
            chase_ok, chase_atr = self._long_chase_ok(row)
            if not chase_ok:
                return None

            quality = self.trend_quality_score(row, Side.LONG)
            if quality < self.cfg.min_trend_quality:
                return None

            return EntrySignal(
                side=Side.LONG,
                entry_price=entry,
                stop_loss=entry * (1.0 - self.cfg.stop_loss_pct),
                take_profit=entry * (1.0 + self.cfg.take_profit_pct),
                trend=trend,
                ema20=float(row["ema20"]),
                ema50=float(row["ema50"]),
                hma16=float(row["hma16"]),
                atr14=float(row["atr14"]),
                adx=float(row["adx"]),
                plus_di=float(row["plus_di"]),
                minus_di=float(row["minus_di"]),
                chop=float(row["chop"]),
                trend_quality=quality,
                ema_separation_atr=float(row["ema_separation_atr"]),
                hma_slope_atr=float(row["hma_slope_atr"]),
                chase_atr=chase_atr,
                reason=(
                    "BULL EMA20/50 trend + HMA16 orange->blue flip + "
                    f"TrendQuality={quality:.0f}/100 "
                    f"(ADX={row['adx']:.1f}, CHOP={row['chop']:.1f}, "
                    f"+DI={row['plus_di']:.1f}, -DI={row['minus_di']:.1f})"
                ),
            )

        if trend == Trend.BEAR and self._flip_down(prev_row, row):
            chase_ok, chase_atr = self._short_chase_ok(row)
            if not chase_ok:
                return None

            quality = self.trend_quality_score(row, Side.SHORT)
            if quality < self.cfg.min_trend_quality:
                return None

            return EntrySignal(
                side=Side.SHORT,
                entry_price=entry,
                stop_loss=entry * (1.0 + self.cfg.stop_loss_pct),
                take_profit=entry * (1.0 - self.cfg.take_profit_pct),
                trend=trend,
                ema20=float(row["ema20"]),
                ema50=float(row["ema50"]),
                hma16=float(row["hma16"]),
                atr14=float(row["atr14"]),
                adx=float(row["adx"]),
                plus_di=float(row["plus_di"]),
                minus_di=float(row["minus_di"]),
                chop=float(row["chop"]),
                trend_quality=quality,
                ema_separation_atr=float(row["ema_separation_atr"]),
                hma_slope_atr=float(row["hma_slope_atr"]),
                chase_atr=chase_atr,
                reason=(
                    "BEAR EMA20/50 trend + HMA16 blue->orange flip + "
                    f"TrendQuality={quality:.0f}/100 "
                    f"(ADX={row['adx']:.1f}, CHOP={row['chop']:.1f}, "
                    f"+DI={row['plus_di']:.1f}, -DI={row['minus_di']:.1f})"
                ),
            )

        return None

    def evaluate_exit(
        self,
        candles: pd.DataFrame,
        position: PositionState,
        current_price: Optional[float] = None,
    ) -> ExitSignal:
        """
        Exit precedence:
          1) Live TP/SL for spike protection.
          2) Opposite HMA16 flip on latest closed 15m candle.

        ADX/CHOP/DMI are entry filters only and do not force exits.
        """
        if current_price is not None:
            px = float(current_price)

            if position.side == Side.LONG:
                if px <= position.stop_loss:
                    return ExitSignal(True, ExitReason.STOP_LOSS, px)
                if px >= position.take_profit:
                    return ExitSignal(True, ExitReason.TAKE_PROFIT, px)
            else:
                if px >= position.stop_loss:
                    return ExitSignal(True, ExitReason.STOP_LOSS, px)
                if px <= position.take_profit:
                    return ExitSignal(True, ExitReason.TAKE_PROFIT, px)

        min_bars = max(self.cfg.hma_len * 2, 5)
        if len(candles) < min_bars:
            return ExitSignal(False)

        df = self.add_indicators(candles)
        prev_row = df.iloc[-2]
        row = df.iloc[-1]

        if position.side == Side.LONG and self._flip_down(prev_row, row):
            return ExitSignal(True, ExitReason.HMA_FLIP, float(row["close"]))

        if position.side == Side.SHORT and self._flip_up(prev_row, row):
            return ExitSignal(True, ExitReason.HMA_FLIP, float(row["close"]))

        return ExitSignal(False)


def build_position_from_signal(
    signal: EntrySignal,
    opened_bar_time: Optional[pd.Timestamp] = None,
) -> PositionState:
    return PositionState(
        side=signal.side,
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        opened_bar_time=opened_bar_time,
    )



# ============================================================
# PORTFOLIO / EXECUTION DEFAULTS
# ============================================================

@dataclass(frozen=True)
class ExecutionConfig:
    max_open_positions: int = 2
    margin_per_position_usd: float = 20.0
    leverage: int = 20
    margin_mode: str = "isolated"
    max_positions_per_symbol: int = 1

    @property
    def notional_per_position_usd(self) -> float:
        return self.margin_per_position_usd * self.leverage

    @property
    def max_total_margin_usd(self) -> float:
        return self.max_open_positions * self.margin_per_position_usd

    @property
    def max_total_notional_usd(self) -> float:
        return self.max_open_positions * self.notional_per_position_usd


@dataclass(frozen=True)
class OpenPosition:
    symbol: str
    side: Side
    entry_price: float
    size: float
    margin_usd: float
    leverage: int


@dataclass(frozen=True)
class OrderPlan:
    allowed: bool
    symbol: str
    side: Optional[Side] = None
    entry_price: Optional[float] = None
    quantity: Optional[float] = None
    notional_usd: Optional[float] = None
    margin_usd: Optional[float] = None
    leverage: Optional[int] = None
    margin_mode: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""


class PortfolioRiskManager:
    """
    Portfolio-level gate for execution.

    Defaults:
      - max 2 open positions total
      - max 1 position per symbol
      - $20 margin per position
      - x20 leverage
      - isolated margin
      - approx. $400 notional per position
    """

    def __init__(self, config: Optional[ExecutionConfig] = None) -> None:
        self.cfg = config or ExecutionConfig()

    def can_open(
        self,
        symbol: str,
        open_positions: list[OpenPosition],
    ) -> tuple[bool, str]:
        if len(open_positions) >= self.cfg.max_open_positions:
            return False, "MAX_OPEN_POSITIONS_REACHED"

        same_symbol = sum(1 for p in open_positions if p.symbol == symbol)
        if same_symbol >= self.cfg.max_positions_per_symbol:
            return False, "SYMBOL_ALREADY_HAS_POSITION"

        return True, "OK"

    def build_order_plan(
        self,
        symbol: str,
        signal: EntrySignal,
        open_positions: list[OpenPosition],
    ) -> OrderPlan:
        allowed, reason = self.can_open(symbol, open_positions)
        if not allowed:
            return OrderPlan(
                allowed=False,
                symbol=symbol,
                reason=reason,
            )

        entry = float(signal.entry_price)
        notional = float(self.cfg.notional_per_position_usd)

        if entry <= 0:
            return OrderPlan(
                allowed=False,
                symbol=symbol,
                reason="INVALID_ENTRY_PRICE",
            )

        # Generic base-asset quantity. Exchange adapter should round this
        # to the instrument's lot size / contract size before order placement.
        quantity = notional / entry

        return OrderPlan(
            allowed=True,
            symbol=symbol,
            side=signal.side,
            entry_price=entry,
            quantity=quantity,
            notional_usd=notional,
            margin_usd=self.cfg.margin_per_position_usd,
            leverage=self.cfg.leverage,
            margin_mode=self.cfg.margin_mode,
            stop_loss=float(signal.stop_loss),
            take_profit=float(signal.take_profit),
            reason="ENTRY_APPROVED",
        )


class HMA16TrendFollowBot:
    """
    Thin orchestration layer combining:
      - HMA16TrendFollowStrategy
      - PortfolioRiskManager

    This still does NOT place exchange orders by itself.
    Your OKX adapter should:
      1) set isolated margin mode
      2) set leverage to x20
      3) round quantity to exchange lot/contract rules
      4) place entry + protective TP/SL
      5) keep HMA16 flip exit active on each closed 15m candle
    """

    def __init__(
        self,
        strategy_config: Optional[StrategyConfig] = None,
        execution_config: Optional[ExecutionConfig] = None,
    ) -> None:
        self.strategy = HMA16TrendFollowStrategy(strategy_config)
        self.risk = PortfolioRiskManager(execution_config)

    def evaluate_symbol(
        self,
        symbol: str,
        candles: pd.DataFrame,
        open_positions: list[OpenPosition],
    ) -> OrderPlan:
        has_symbol_position = any(p.symbol == symbol for p in open_positions)

        signal = self.strategy.generate_entry(
            candles,
            has_open_position=has_symbol_position,
        )

        if signal is None:
            return OrderPlan(
                allowed=False,
                symbol=symbol,
                reason="NO_VALID_ENTRY_SIGNAL",
            )

        return self.risk.build_order_plan(
            symbol=symbol,
            signal=signal,
            open_positions=open_positions,
        )


DEFAULT_STRATEGY_CONFIG = StrategyConfig()
DEFAULT_EXECUTION_CONFIG = ExecutionConfig()


if __name__ == "__main__":
    print("HMA16 Trend-Follow v2 (ADX + CHOP + DMI) module loaded.")
