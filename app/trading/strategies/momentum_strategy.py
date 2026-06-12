"""
Momentum composite strategy: MACD + RSI + ROC + RVOL scoring.

Each indicator casts a vote (-1 / 0 / +1).
Score ≥ +3  → BUY
Score ≤ -3  → SELL

Votes:
  RSI  — direction + zone (+2 / -2 possible)
  MACD — histogram sign + direction (+2 / -2 possible)
  ROC  — sign + magnitude (+1 / -1)
  RVOL — volume spike multiplies confidence (not a vote, used for conf)
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class MomentumStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.rsi_period   = self.params.get("rsi_period",   14)
        self.macd_fast    = self.params.get("macd_fast",    12)
        self.macd_slow    = self.params.get("macd_slow",    26)
        self.macd_sig     = self.params.get("macd_signal",  9)
        self.roc_period   = self.params.get("roc_period",   10)
        self.rvol_period  = self.params.get("rvol_period",  20)
        self.buy_score    = self.params.get("buy_score",    3)
        self.sell_score   = self.params.get("sell_score",   -3)
        self.position_pct = self.params.get("position_pct", 0.08)

    async def analyze(self, candles: list, current_price: float) -> Signal:
        min_len = self.macd_slow + self.macd_sig + self.roc_period + 5
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        closes  = [c.close  for c in candles]
        volumes = [c.volume for c in candles]

        # ── RSI ─────────────────────────────────────────────────────────
        rsi_arr  = self.rsi(closes, self.rsi_period)
        curr_rsi = float(rsi_arr[-1])
        prev_rsi = float(rsi_arr[-2])
        if np.isnan(curr_rsi): curr_rsi = 50.0; prev_rsi = 50.0

        rsi_vote = 0
        if curr_rsi < 50 and curr_rsi > prev_rsi:   rsi_vote += 1   # rising from low
        if curr_rsi < 40:                            rsi_vote += 1   # oversold zone
        if curr_rsi > 50 and curr_rsi < prev_rsi:   rsi_vote -= 1   # falling from high
        if curr_rsi > 60:                            rsi_vote -= 1   # overbought zone

        # ── MACD ────────────────────────────────────────────────────────
        _, _, hist = self.macd(closes, self.macd_fast, self.macd_slow, self.macd_sig)
        curr_hist = float(hist[-1]); prev_hist = float(hist[-2])
        if np.isnan(curr_hist): curr_hist = 0.0; prev_hist = 0.0

        macd_vote = 0
        if curr_hist > 0:                macd_vote += 1   # above zero
        if curr_hist > prev_hist:        macd_vote += 1   # rising
        if curr_hist < 0:                macd_vote -= 1   # below zero
        if curr_hist < prev_hist:        macd_vote -= 1   # falling

        # ── ROC (Rate of Change) ─────────────────────────────────────────
        c_arr    = np.array(closes)
        roc      = 0.0
        if len(c_arr) > self.roc_period and c_arr[-(self.roc_period + 1)] > 0:
            roc = (c_arr[-1] - c_arr[-(self.roc_period + 1)]) / c_arr[-(self.roc_period + 1)] * 100

        roc_vote = 1 if roc > 0.1 else (-1 if roc < -0.1 else 0)

        # ── RVOL ─────────────────────────────────────────────────────────
        vol_arr = np.array(volumes)
        vol_ma  = float(np.mean(vol_arr[-(self.rvol_period + 1):-1])) if len(vol_arr) > self.rvol_period else float(np.mean(vol_arr))
        rvol    = float(vol_arr[-1]) / max(vol_ma, 1e-8)

        # ── Scoring ──────────────────────────────────────────────────────
        score = rsi_vote + macd_vote + roc_vote
        # RVOL boosts confidence but doesn't vote
        conf_base = abs(score) / 5.0
        rvol_boost = min(0.3, (rvol - 1.0) * 0.1) if rvol > 1.0 else 0.0
        confidence = min(1.0, conf_base + rvol_boost + 0.2)

        detail = (f"RSI={curr_rsi:.1f}({rsi_vote:+d}) "
                  f"MACD={curr_hist:.5f}({macd_vote:+d}) "
                  f"ROC={roc:.2f}%({roc_vote:+d}) "
                  f"RVOL={rvol:.1f}x → score={score:+d}")

        if score >= self.buy_score:
            return Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct,
                reason=f"[Momentum] BUY score={score:+d} | {detail}",
                confidence=confidence,
                metadata={"score": score, "rsi": curr_rsi, "macd_hist": curr_hist,
                          "roc": roc, "rvol": rvol},
            )

        if score <= self.sell_score:
            return Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct,
                reason=f"[Momentum] SELL score={score:+d} | {detail}",
                confidence=confidence,
                metadata={"score": score, "rsi": curr_rsi, "macd_hist": curr_hist,
                          "roc": roc, "rvol": rvol},
            )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[Momentum] score={score:+d} | {detail}",
            metadata={"score": score, "rsi": curr_rsi, "macd_hist": curr_hist,
                      "roc": roc, "rvol": rvol},
        )
