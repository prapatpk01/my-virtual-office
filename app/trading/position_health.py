"""
Position Health Monitor — algorithmic re-evaluation of open positions.

Runs every MONITOR_INTERVAL seconds (default 3 min) on a separate asyncio task.
Fetches 5m candles + 1h/4h MTF for each symbol with an open position, then
computes a weighted health score (0–100) against the position direction.

Score → Label → Action (4-level)
  85–100 BULL        Hold / extend TP2 one step toward next level (→3.0R)
  50–84  NEUTRAL     Hold — do nothing
  40–49  WEAK        close after 3-cycle confirm (soft fade, may recover)
  0–39   STRONG_WEAK close NOW, no confirm (sharp reversal / V-shape)

Indicators (total 100) — split slow-structure / fast-momentum so a sudden
V-shape (5m collapse before the 4H/1H trend flips) drops the score fast:
  SLOW structure (lags, 54)
    20  4H EMA20 vs EMA50 aligned
    15  1H EMA20 vs EMA50 aligned
     8  MTF composite bias > ±30
     7  ADX(14,5m) > 20
     4  1H price above EMA20 (structural support)
  FAST momentum (reacts in 1-2 bars, 46)
     8  5m close vs EMA9 aligned
     7  5m close vs EMA20 aligned
     5  RSI(14,5m) not at opposite extreme
    10  ROC(3,5m) velocity favorable          ← V-shape detector (was 18)
     8  MACD histogram (5m) favorable         ← leads EMA cross (was 12)
     5  Volume expansion (vol > MA20 × 1.15)  ← breakout momentum
     3  EMA5 > EMA20 micro-stack on 5m        ← short-term alignment
  −20  PENALTY: accelerating reversal (3 adverse closes + range expanding)
       → a sharp V-shape can pull a 95 score to ~35 (STRONG_WEAK = close now)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .connectors.base import BaseConnector

logger = logging.getLogger("pos_health")

# ── Lightweight indicator helpers ─────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))

def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1/n, adjust=False).mean()

def _tr(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"]-df["low"], (df["high"]-pc).abs(), (df["low"]-pc).abs()], axis=1).max(axis=1)

def _adx(df: pd.DataFrame, n: int = 14) -> float:
    up  = df["high"].diff()
    dn  = -df["low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr_rma  = _rma(_tr(df), n)
    pdi = 100 * _rma(pd.Series(pdm, index=df.index), n) / tr_rma
    mdi = 100 * _rma(pd.Series(mdm, index=df.index), n) / tr_rma
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return float(_rma(dx.fillna(0), n).iloc[-1])


@dataclass
class HealthResult:
    score: float         # 0–100
    label: str           # BULL / NEUTRAL / CAUTION / WEAK
    action: str          # HOLD / EXTEND_TP / CLOSE
    details: dict        # per-indicator breakdown for logging / Telegram

    def __str__(self) -> str:
        parts = [f"{k}={'✓' if v else '✗'}" for k, v in self.details.items()]
        return f"[{self.label} {self.score:.0f}%] {' '.join(parts)}"


class PositionHealthMonitor:
    """
    Fetches candles and computes position health on demand.
    Designed to be called from the bot's monitor loop.
    """

    TP1_LADDER = [0.5, 0.8, 1.2]  # R-multiples for TP1 advance (max 2 raises, cap 1.2R)
    TP_LADDER  = [1.2, 1.5, 2.0, 2.5, 3.0]  # R-multiples for TP2 advance (after TP1 hit)

    def __init__(self, connector: "BaseConnector"):
        self.connector = connector

    async def check(
        self,
        symbol: str,
        side: str,                  # 'long' | 'short'
        entry_price: float,
        oneR: float,                # original SL distance (1R)
        current_tp2: float,         # current TP2 price
        candles_5m: Optional[list] = None,   # pre-fetched (avoids double fetch)
        candles_1h: Optional[list] = None,
        candles_4h: Optional[list] = None,
    ) -> HealthResult:
        """
        Returns a HealthResult with score, label, and recommended action.
        Fetches 5m/1h/4h candles if not provided.
        """
        try:
            if candles_5m is None:
                candles_5m = await self.connector.fetch_ohlcv(symbol, "5m",  limit=120)
            if candles_1h is None:
                candles_1h = await self.connector.fetch_ohlcv(symbol, "1h",  limit=80)
            if candles_4h is None:
                candles_4h = await self.connector.fetch_ohlcv(symbol, "4h",  limit=60)
        except Exception as e:
            logger.warning("[HealthMon] %s candle fetch failed: %s", symbol, e)
            return HealthResult(score=60.0, label="NEUTRAL", action="HOLD",
                                details={"error": str(e)})

        df5  = _to_df(candles_5m)
        df1h = _to_df(candles_1h)
        df4h = _to_df(candles_4h)

        if len(df5) < 30 or len(df1h) < 30 or len(df4h) < 15:
            logger.debug("[HealthMon] %s not enough candles — neutral", symbol)
            return HealthResult(score=60.0, label="NEUTRAL", action="HOLD", details={})

        is_long = side == "long"
        score, details = self._score(df5, df1h, df4h, is_long)

        label  = _classify(score)
        action = _action(label, score)

        result = HealthResult(score=score, label=label, action=action, details=details)
        logger.info("[HealthMon] %s %s  %s", symbol, side.upper(), result)
        return result

    def _score(
        self, df5: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame, is_long: bool
    ) -> tuple[float, dict]:
        """Compute weighted health score and per-indicator breakdown."""
        details: dict[str, bool] = {}
        total_score = 0.0

        # ── SLOW structure (lags a V-shape) — total 54 ───────────────────────
        # ── 4H trend (EMA20 vs EMA50) ────────────────────────────────────────
        ef4 = _ema(df4h["close"], 20).iloc[-1]
        es4 = _ema(df4h["close"], 50).iloc[-1]
        ok_4h = (ef4 > es4) if is_long else (ef4 < es4)
        details["4H_trend"] = bool(ok_4h)
        if ok_4h: total_score += 20

        # ── 1H trend (EMA20 vs EMA50) ────────────────────────────────────────
        ef1 = _ema(df1h["close"], 20).iloc[-1]
        es1 = _ema(df1h["close"], 50).iloc[-1]
        ok_1h = (ef1 > es1) if is_long else (ef1 < es1)
        details["1H_trend"] = bool(ok_1h)
        if ok_1h: total_score += 15

        # ── 1H price above EMA20 (structural support) ────────────────────────
        price_1h = float(df1h["close"].iloc[-1])
        ok_1h_price = (price_1h > ef1) if is_long else (price_1h < ef1)
        details["1H_above_EMA20"] = bool(ok_1h_price)
        if ok_1h_price: total_score += 4

        # ── MTF composite bias (relaxed: ±30) ────────────────────────────────
        mtf_score = _mtf_bias(df5["close"], df1h["close"], df4h["close"])
        ok_bias = (mtf_score > 30) if is_long else (mtf_score < -30)
        details["MTF_bias"] = bool(ok_bias)
        details["_mtf_bias_val"] = round(float(mtf_score), 1)
        if ok_bias: total_score += 8

        # ── ADX (relaxed: > 20) ──────────────────────────────────────────────
        adx_val = _adx(df5.tail(60), 14)
        ok_adx = adx_val > 20
        details["ADX>20"] = bool(ok_adx)
        details["_adx>20_val"] = round(adx_val, 1)
        if ok_adx: total_score += 7

        # ── FAST momentum (reacts within 1-2 bars → catches V-shape) — total 50
        price = df5["close"].iloc[-1]

        # 5m close vs EMA9 / EMA20
        ema9  = _ema(df5["close"], 9).iloc[-1]
        ok_ema9 = (price > ema9) if is_long else (price < ema9)
        details["Close>EMA9"] = bool(ok_ema9)
        if ok_ema9: total_score += 8

        ema20 = _ema(df5["close"], 20).iloc[-1]
        ok_ema20 = (price > ema20) if is_long else (price < ema20)
        details["Close>EMA20"] = bool(ok_ema20)
        if ok_ema20: total_score += 7

        # RSI not at opposite extreme
        rsi_val = float(_rsi(df5["close"], 14).iloc[-1])
        ok_rsi = (rsi_val < 75) if is_long else (rsi_val > 25)
        details["RSI_ok"] = bool(ok_rsi)
        details["_rsi_ok_val"] = round(rsi_val, 1)
        if ok_rsi: total_score += 5

        # ── EMA5 micro-stack (EMA5 > EMA20 for long = short-term aligned) ─────
        ema5_5m = float(_ema(df5["close"], 5).iloc[-1])
        ok_ema5_align = (ema5_5m > ema20) if is_long else (ema5_5m < ema20)
        details["EMA5_align"] = bool(ok_ema5_align)
        if ok_ema5_align: total_score += 3

        # ── Volume expansion (vol > MA20 × 1.15 = breakout momentum) ─────────
        if len(df5) >= 20:
            vol_ma20 = float(df5["volume"].rolling(20).mean().iloc[-1])
            vol_now  = float(df5["volume"].iloc[-1])
            ok_vol_exp = (vol_now > vol_ma20 * 1.15) if vol_ma20 > 0 else False
        else:
            ok_vol_exp = False
        details["Vol_expansion"] = bool(ok_vol_exp)
        if ok_vol_exp: total_score += 5

        # ── V-shape detector 1: ROC(3) on 5m (price velocity over 15 min) ─────
        if len(df5) >= 4:
            c0 = float(df5["close"].iloc[-1]); c3 = float(df5["close"].iloc[-4])
            roc3 = (c0 - c3) / c3 * 100 if c3 else 0.0
        else:
            roc3 = 0.0
        ok_roc = (roc3 > 0) if is_long else (roc3 < 0)
        details["ROC3_favorable"] = bool(ok_roc)
        details["_roc3_val"] = round(roc3, 2)
        if ok_roc: total_score += 10

        # ── V-shape detector 2: 5m MACD histogram (momentum, leads EMA cross) ─
        macd_line = _ema(df5["close"], 12) - _ema(df5["close"], 26)
        macd_sig  = _ema(macd_line, 9)
        hist = float((macd_line - macd_sig).iloc[-1])
        ok_macd = (hist > 0) if is_long else (hist < 0)
        details["MACD_hist_favorable"] = bool(ok_macd)
        details["_macd_hist"] = round(hist, 4)
        if ok_macd: total_score += 8

        # ── V-shape detector 3: accelerating reversal PENALTY ────────────────
        # The V-shape signature: 3 consecutive closes against + last range
        # expanding. This is a strong, specific reversal pattern, so it is a
        # penalty (−20) rather than a small award: a sharp dump pulls a healthy
        # score below 40 (STRONG_WEAK → close NOW), while gentle pullbacks that
        # lack the 3-consecutive-expanding pattern are untouched.
        accel_reversal = False
        if len(df5) >= 4:
            cl = df5["close"].values; hi = df5["high"].values; lo = df5["low"].values
            if is_long:
                three_down = cl[-1] < cl[-2] < cl[-3] < cl[-4]
            else:
                three_down = cl[-1] > cl[-2] > cl[-3] > cl[-4]
            rng_now  = hi[-1] - lo[-1]
            rng_prev = hi[-2] - lo[-2]
            accel_reversal = bool(three_down and rng_now > rng_prev)
        details["No_accel_reversal"] = (not accel_reversal)
        if accel_reversal: total_score -= 20

        return max(0.0, total_score), details

    def next_tp1_level(self, entry: float, oneR: float, current_tp1: float, side: str) -> Optional[float]:
        """
        Before TP1 hit: advance TP1 one rung up TP1_LADDER (max 1.5R).
        Returns None if already at 1.5R cap or no next level exists.
        """
        is_long = side == "long"
        for r in self.TP1_LADDER:
            tp = entry + r * oneR if is_long else entry - r * oneR
            if is_long and tp > current_tp1 + oneR * 0.05:
                return round(tp, 4)
            if not is_long and tp < current_tp1 - oneR * 0.05:
                return round(tp, 4)
        return None  # already at 1.5R cap

    def next_tp_level(self, entry: float, oneR: float, current_tp: float, side: str) -> Optional[float]:
        """
        After TP1 hit: advance TP2 one rung up TP_LADDER (max 3.0R).
        Returns None if already at max.
        """
        is_long = side == "long"
        for r in self.TP_LADDER:
            tp = entry + r * oneR if is_long else entry - r * oneR
            if is_long and tp > current_tp + oneR * 0.1:
                return round(tp, 4)
            if not is_long and tp < current_tp - oneR * 0.1:
                return round(tp, 4)
        return None  # already at 3.0R cap


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_df(candles: list) -> pd.DataFrame:
    """Convert OHLCV list-of-objects or list-of-dicts to DataFrame."""
    if not candles:
        return pd.DataFrame()
    first = candles[0]
    if hasattr(first, "close"):
        rows = [{"open": c.open, "high": c.high, "low": c.low,
                 "close": c.close, "volume": c.volume} for c in candles]
    elif isinstance(first, dict):
        rows = candles
    else:
        rows = [{"open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5]}
                for c in candles if len(c) >= 6]
    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df: df[col] = df[col].astype(float)
    return df


def _mtf_bias(s5: pd.Series, s1h: pd.Series, s4h: pd.Series) -> float:
    """Composite MTF bias −100…+100 using EMA20/50 cross + RSI."""
    def _vote(s: pd.Series) -> int:
        ef = _ema(s, 20).iloc[-1]; es = _ema(s, 50).iloc[-1]
        r  = float(_rsi(s, 14).iloc[-1])
        return (1 if s.iloc[-1] > ef else -1) + (1 if ef > es else -1) + \
               (1 if r > 55 else (-1 if r < 45 else 0))
    votes = [_vote(s5), _vote(s1h), _vote(s4h)]
    return round(sum(votes) / (len(votes) * 3) * 100, 1)


def _classify(score: float) -> str:
    """
    4-level health:
      BULL        85-100  → extend TP (momentum strong)
      NEUTRAL     50-84   → hold
      WEAK        25-49   → close after N-cycle confirm (fade, may recover)
      STRONG_WEAK 0-24    → close NOW, no confirm (sharp reversal / V-shape)
    Lowered STRONG_WEAK cutoff 40→25: only a sharper reversal triggers the
    immediate no-confirm close; scores 25-49 get the WEAK 3-cycle grace so
    moderate dips aren't cut prematurely.
    """
    if score >= 85: return "BULL"
    if score >= 50: return "NEUTRAL"
    if score >= 25: return "WEAK"
    return "STRONG_WEAK"


def _action(label: str, score: float) -> str:
    if label == "BULL":        return "EXTEND_TP"
    if label == "WEAK":        return "CLOSE"   # bot delays via weak-confirm
    if label == "STRONG_WEAK": return "CLOSE"   # bot closes immediately
    return "HOLD"
