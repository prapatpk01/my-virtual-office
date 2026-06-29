"""Momentum Score Strategy — composite score from RSI, MACD, and EMA trend.

Scores each indicator and combines into a buy/sell threshold.

Public interface used by backtest_50usd.py / tune_1h.py:
    _build_signals(candles) → (buy_arr, sell_arr, score_arr, rsi_arr, atr_arr)
    attributes: macd_slow, macd_sig, ema_len, sl_atr_mult, rr_ratio
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class MomentumScoreStrategy(BaseStrategy):
    """Composite momentum score: RSI + MACD + EMA direction."""

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.rsi_len      = int(self.params.get("rsi_len",     14))
        self.ema_len      = int(self.params.get("ema_len",     50))
        self.macd_fast    = int(self.params.get("macd_fast",   12))
        self.macd_slow    = int(self.params.get("macd_slow",   26))
        self.macd_sig     = int(self.params.get("macd_sig",     9))
        self.threshold    = int(self.params.get("threshold",    4))   # points to trigger
        self.sl_atr_mult  = float(self.params.get("sl_atr_mult", 2.0))
        self.rr_ratio     = float(self.params.get("rr_ratio",    1.5))
        self.position_pct = float(self.params.get("position_pct", 0.05))

    # ------------------------------------------------------------------ #

    def _build_signals(self, candles: list):
        """Build signal arrays over the full candle history.

        Returns
        -------
        buy_arr   : np.ndarray[bool]
        sell_arr  : np.ndarray[bool]
        score_arr : np.ndarray[float]  — composite momentum score (-6 to +6)
        rsi_arr   : np.ndarray[float]
        atr_arr   : np.ndarray[float]  — ATR(14)
        """
        n       = len(candles)
        closes  = [c.close for c in candles]

        rsi_arr  = self.rsi(closes, self.rsi_len)
        ema_arr  = self.ema(closes, self.ema_len)
        ml, sl_line, hist = self.macd(closes, self.macd_fast, self.macd_slow, self.macd_sig)
        atr_arr  = np.array(self.atr(candles, 14), dtype=float)

        ml       = np.asarray(ml,      dtype=float)
        sl_line  = np.asarray(sl_line, dtype=float)
        hist_arr = np.asarray(hist,    dtype=float)

        score_arr = np.full(n, np.nan)
        buy_arr   = np.zeros(n, dtype=bool)
        sell_arr  = np.zeros(n, dtype=bool)

        for i in range(1, n):
            rsi  = float(rsi_arr[i])
            ema  = float(ema_arr[i])
            macd = float(ml[i])
            sig  = float(sl_line[i])
            atr  = float(atr_arr[i])
            close = closes[i]

            if any(np.isnan(v) for v in [rsi, ema, macd, sig]):
                continue

            # RSI score: +2 if < 40 (oversold), +1 if 40-50, -1 if 50-60, -2 if > 60
            if rsi < 40:
                rsi_score = 2
            elif rsi < 50:
                rsi_score = 1
            elif rsi < 60:
                rsi_score = -1
            else:
                rsi_score = -2

            # EMA score: +2 if price well above, +1 above, -1 below, -2 well below
            gap_pct = (close - ema) / max(ema, 1e-8) * 100
            if gap_pct > 1.0:
                ema_score = 2
            elif gap_pct > 0:
                ema_score = 1
            elif gap_pct > -1.0:
                ema_score = -1
            else:
                ema_score = -2

            # MACD score: +2 if histogram positive and rising, +1 if positive, etc.
            prev_hist = float(hist_arr[i - 1]) if not np.isnan(hist_arr[i - 1]) else 0.0
            curr_hist = float(hist_arr[i])
            if curr_hist > 0 and curr_hist > prev_hist:
                macd_score = 2
            elif curr_hist > 0:
                macd_score = 1
            elif curr_hist < 0 and curr_hist < prev_hist:
                macd_score = -2
            else:
                macd_score = -1

            score = rsi_score + ema_score + macd_score   # range: -6 to +6
            score_arr[i] = score

            # Previous score for direction change detection
            prev_score = float(score_arr[i - 1]) if not np.isnan(score_arr[i - 1]) else 0

            buy_arr[i]  = score >= self.threshold  and prev_score < self.threshold
            sell_arr[i] = score <= -self.threshold and prev_score > -self.threshold

        return buy_arr, sell_arr, score_arr, rsi_arr, atr_arr

    # ------------------------------------------------------------------ #

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = self.macd_slow + self.macd_sig + self.ema_len + 5
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        buy_arr, sell_arr, score_arr, rsi_arr, atr_arr = self._build_signals(candles)

        curr_score = float(score_arr[-1]) if not np.isnan(score_arr[-1]) else 0.0
        curr_rsi   = float(rsi_arr[-1])   if not np.isnan(rsi_arr[-1])   else 50.0
        curr_atr   = float(atr_arr[-1])   if not np.isnan(atr_arr[-1])   else 0.0
        conf       = min(1.0, abs(curr_score) / 6.0)

        if buy_arr[-1]:
            sl = current_price - self.sl_atr_mult * curr_atr
            tp = current_price + self.sl_atr_mult * curr_atr * self.rr_ratio
            return Signal(
                SignalType.BUY, self.symbol, current_price, self.position_pct,
                f"[MomScore] Score={curr_score:.0f} RSI={curr_rsi:.1f}",
                confidence=conf,
                metadata={"score": curr_score, "rsi": curr_rsi, "atr": curr_atr, "sl": sl, "tp": tp},
            )
        if sell_arr[-1]:
            sl = current_price + self.sl_atr_mult * curr_atr
            tp = current_price - self.sl_atr_mult * curr_atr * self.rr_ratio
            return Signal(
                SignalType.SELL, self.symbol, current_price, self.position_pct,
                f"[MomScore] Score={curr_score:.0f} RSI={curr_rsi:.1f}",
                confidence=conf,
                metadata={"score": curr_score, "rsi": curr_rsi, "atr": curr_atr, "sl": sl, "tp": tp},
            )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[MomScore] Score={curr_score:.0f} (threshold=±{self.threshold})",
            metadata={"score": curr_score, "rsi": curr_rsi},
        )
