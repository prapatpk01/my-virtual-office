"""RSI + MACD strategy with 3-TF MTF Bias gate (15m / 1H / 4H)."""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class RSIMACDStrategy(BaseStrategy):
    """
    Entry conditions (base TF — 15m by default):
      BUY:  RSI < oversold AND MACD histogram rising/crossover  |  OR strong MACD cross + RVOL
      SELL: RSI > overbought AND MACD histogram falling/crossover | OR strong MACD cross + RVOL

    MTF Bias gate (always applied when mtf_candles provided):
      Scores 15m / 1H / 4H independently using EMA20, EMA50, RSI.
      Weighted composite (w=1/2/3) → -100..+100 %.
      BUY  passes only when composite > 0  (majority of TFs bullish).
      SELL passes only when composite < 0  (majority of TFs bearish).
      If mtf_candles unavailable → gate skipped (graceful fallback).
    """

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.rsi_period     = self.params.get("rsi_period",    14)
        self.rsi_oversold   = self.params.get("rsi_oversold",  42)
        self.rsi_overbought = self.params.get("rsi_overbought", 58)
        self.macd_fast      = self.params.get("macd_fast",     12)
        self.macd_slow      = self.params.get("macd_slow",     26)
        self.macd_signal    = self.params.get("macd_signal",    9)
        self.position_pct   = self.params.get("position_pct",  0.1)
        # MTF gate threshold: absolute comp_pct must exceed this to confirm direction
        self.mtf_threshold  = self.params.get("mtf_threshold", 0.0)

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        closes  = [c.close  for c in candles]
        volumes = [c.volume for c in candles]
        if len(closes) < self.macd_slow + self.macd_signal + 5:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        # ── Base-TF indicators ───────────────────────────────────────────
        rsi_arr = self.rsi(closes, self.rsi_period)
        macd_line, signal_line, histogram = self.macd(
            closes, self.macd_fast, self.macd_slow, self.macd_signal
        )

        curr_rsi  = float(rsi_arr[-1])
        prev_hist = float(histogram[-2])
        curr_hist = float(histogram[-1])

        if np.isnan(curr_rsi) or np.isnan(curr_hist) or np.isnan(prev_hist):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Indicator NaN")

        vol_arr = np.array(volumes)
        vol_ma  = float(np.mean(vol_arr[-20:])) if len(vol_arr) >= 20 else float(np.mean(vol_arr))
        rvol    = float(vol_arr[-1]) / max(vol_ma, 1e-8)

        hist_rising     = curr_hist > prev_hist
        hist_falling    = curr_hist < prev_hist
        macd_cross_up   = prev_hist <= 0 < curr_hist
        macd_cross_down = prev_hist >= 0 > curr_hist

        # ── MTF Bias (15m / 1H / 4H) ────────────────────────────────────
        # compute_mtf_bias uses candles as 15m, mtf_candles["1h"] and ["4h"]
        mtf_available = bool(mtf_candles and (mtf_candles.get("1h") or mtf_candles.get("4h")))
        if mtf_available:
            mtf_comp, mtf_label = self.compute_mtf_bias(candles, mtf_candles)
            mtf_bull = mtf_comp > self.mtf_threshold
            mtf_bear = mtf_comp < -self.mtf_threshold
        else:
            mtf_comp, mtf_label = 0.0, "MTF N/A"
            mtf_bull = mtf_bear = True  # gate open when no MTF data

        # ── BUY conditions ───────────────────────────────────────────────
        buy1 = curr_rsi < self.rsi_oversold and macd_cross_up
        buy2 = curr_rsi < self.rsi_oversold and hist_rising and rvol >= 1.3
        buy3 = macd_cross_up and curr_rsi < 55 and rvol >= 1.5

        if buy1 or buy2 or buy3:
            if not mtf_bull:
                # Signal exists but MTF rejects it
                return Signal(
                    SignalType.HOLD, self.symbol, current_price, 0,
                    f"[RSI+MACD] BUY blocked by MTF [{mtf_label}] RSI={curr_rsi:.1f}",
                    metadata={"rsi": curr_rsi, "macd_hist": curr_hist,
                              "rvol": rvol, "mtf_comp": round(mtf_comp, 1)},
                )
            reason_tag = ("RSI+MACD crossover" if buy1
                          else f"RSI oversold+RVOL={rvol:.1f}x" if buy2
                          else f"MACD crossover+RVOL={rvol:.1f}x")
            conf = min(1.0, (self.rsi_oversold - curr_rsi) / self.rsi_oversold + 0.3)
            return Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct,
                reason=f"[RSI+MACD] {reason_tag} RSI={curr_rsi:.1f} | MTF [{mtf_label}]",
                confidence=conf,
                metadata={"rsi": curr_rsi, "macd_hist": curr_hist,
                          "rvol": rvol, "mtf_comp": round(mtf_comp, 1)},
            )

        # ── SELL conditions ──────────────────────────────────────────────
        sell1 = curr_rsi > self.rsi_overbought and macd_cross_down
        sell2 = curr_rsi > self.rsi_overbought and hist_falling and rvol >= 1.3
        sell3 = macd_cross_down and curr_rsi > 45 and rvol >= 1.5

        if sell1 or sell2 or sell3:
            if not mtf_bear:
                return Signal(
                    SignalType.HOLD, self.symbol, current_price, 0,
                    f"[RSI+MACD] SELL blocked by MTF [{mtf_label}] RSI={curr_rsi:.1f}",
                    metadata={"rsi": curr_rsi, "macd_hist": curr_hist,
                              "rvol": rvol, "mtf_comp": round(mtf_comp, 1)},
                )
            reason_tag = ("RSI+MACD crossover" if sell1
                          else f"RSI overbought+RVOL={rvol:.1f}x" if sell2
                          else f"MACD crossover+RVOL={rvol:.1f}x")
            conf = min(1.0, (curr_rsi - self.rsi_overbought) / (100 - self.rsi_overbought) + 0.3)
            return Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct,
                reason=f"[RSI+MACD] {reason_tag} RSI={curr_rsi:.1f} | MTF [{mtf_label}]",
                confidence=conf,
                metadata={"rsi": curr_rsi, "macd_hist": curr_hist,
                          "rvol": rvol, "mtf_comp": round(mtf_comp, 1)},
            )

        zone = "oversold" if curr_rsi < 45 else "overbought" if curr_rsi > 55 else "neutral"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[RSI+MACD] RSI={curr_rsi:.1f} ({zone}) hist={curr_hist:.5f} "
            f"RVOL={rvol:.1f}x | MTF [{mtf_label}]",
            metadata={"rsi": curr_rsi, "macd_hist": curr_hist,
                      "rvol": rvol, "mtf_comp": round(mtf_comp, 1)},
        )
