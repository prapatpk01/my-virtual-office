"""
MACD + EMA Voting Strategy — 2/3 confirmation system on 15m timeframe.

Three independent voters:
  A) HMA15 gate   — close > HMA15 → BUY vote;  close < HMA15 → SELL vote
  B) EMA9/SMA21   — EMA9  > SMA21  → BUY vote;  EMA9  < SMA21  → SELL vote
  C) MACD         — MACD  > Signal → BUY vote;  MACD  < Signal → SELL vote

Signal fired when 2 or 3 voters agree:
  2/3 → confidence 0.65
  3/3 → confidence 0.85
  +0.10 bonus if a crossover just happened (B or C)
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class MACDEMAStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.hma_period = self.params.get("hma_period",  15)
        self.ema_fast   = self.params.get("ema_fast",     9)
        self.sma_slow   = self.params.get("sma_slow",    21)
        self.macd_fast  = self.params.get("macd_fast",   12)
        self.macd_slow  = self.params.get("macd_slow",   26)
        self.macd_sig   = self.params.get("macd_signal",  9)

    async def analyze(self, candles: list, current_price: float) -> Signal:
        min_len = self.macd_slow + self.macd_sig + self.hma_period + 5
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        closes = [c.close for c in candles]

        hma15      = self.hma(closes, self.hma_period)
        ema9       = self.ema(closes, self.ema_fast)
        sma21      = self.sma(closes, self.sma_slow)
        macd_line, signal_line, _ = self.macd(
            closes, self.macd_fast, self.macd_slow, self.macd_sig
        )

        p     = current_price
        hma   = float(hma15[-1])
        e9_c  = float(ema9[-1]);     e9_p  = float(ema9[-2])
        s21_c = float(sma21[-1]);    s21_p = float(sma21[-2])
        ml_c  = float(macd_line[-1]); ml_p = float(macd_line[-2])
        sl_c  = float(signal_line[-1]); sl_p = float(signal_line[-2])

        if any(np.isnan(v) for v in [hma, e9_c, s21_c, ml_c, sl_c]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Indicator NaN")

        # ── 3 voters (+1 = BUY, -1 = SELL, 0 = neutral) ─────────────
        vote_a = 1 if p > hma   else (-1 if p < hma   else 0)  # HMA15
        vote_b = 1 if e9_c > s21_c else (-1 if e9_c < s21_c else 0)  # EMA9/SMA21
        vote_c = 1 if ml_c > sl_c  else (-1 if ml_c < sl_c  else 0)  # MACD

        buy_votes  = sum(1 for v in [vote_a, vote_b, vote_c] if v ==  1)
        sell_votes = sum(1 for v in [vote_a, vote_b, vote_c] if v == -1)

        # Crossover detection (confidence bonus + reason label)
        b_cross_up   = e9_p <= s21_p and e9_c > s21_c
        b_cross_down = e9_p >= s21_p and e9_c < s21_c
        c_cross_up   = ml_p <= sl_p  and ml_c > sl_c
        c_cross_down = ml_p >= sl_p  and ml_c < sl_c

        def _build_reason(direction: str, votes: int) -> tuple[str, float]:
            tags = []
            if vote_a == (1 if direction == "buy" else -1):
                tags.append(f"HMA15={'above' if direction=='buy' else 'below'}")
            if vote_b == (1 if direction == "buy" else -1):
                cross = " ✚cross" if (b_cross_up if direction == "buy" else b_cross_down) else ""
                tags.append(f"EMA9{'>' if direction=='buy' else '<'}SMA21{cross}")
            if vote_c == (1 if direction == "buy" else -1):
                cross = " ✚cross" if (c_cross_up if direction == "buy" else c_cross_down) else ""
                tags.append(f"MACD{'>' if direction=='buy' else '<'}Sig{cross}")

            conf = 0.85 if votes == 3 else 0.65
            if (direction == "buy"  and (b_cross_up  or c_cross_up)) or \
               (direction == "sell" and (b_cross_down or c_cross_down)):
                conf = min(conf + 0.10, 0.95)

            return f"[{votes}/3] {' + '.join(tags)}", conf

        if buy_votes >= 2:
            reason, conf = _build_reason("buy", buy_votes)
            return Signal(
                type=SignalType.BUY, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=reason,
                metadata={"hma15": hma, "ema9": e9_c, "sma21": s21_c,
                          "macd": ml_c, "signal": sl_c,
                          "votes": buy_votes},
            )

        if sell_votes >= 2:
            reason, conf = _build_reason("sell", sell_votes)
            return Signal(
                type=SignalType.SELL, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=reason,
                metadata={"hma15": hma, "ema9": e9_c, "sma21": s21_c,
                          "macd": ml_c, "signal": sl_c,
                          "votes": sell_votes},
            )

        # HOLD — show vote breakdown for transparency
        votes_str = (
            f"A={'B' if vote_a==1 else 'S' if vote_a==-1 else '-'} "
            f"B={'B' if vote_b==1 else 'S' if vote_b==-1 else '-'} "
            f"C={'B' if vote_c==1 else 'S' if vote_c==-1 else '-'}"
        )
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[HOLD] {votes_str} | HMA15={hma:.2f}",
            metadata={"hma15": hma, "ema9": e9_c, "sma21": s21_c,
                      "macd": ml_c, "signal": sl_c,
                      "buy_votes": buy_votes, "sell_votes": sell_votes},
        )
