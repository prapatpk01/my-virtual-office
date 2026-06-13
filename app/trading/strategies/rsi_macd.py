"""RSI + MACD combined strategy — relaxed thresholds for more frequent signals."""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class RSIMACDStrategy(BaseStrategy):
    """
    Buy:  RSI < oversold (42) AND MACD histogram rising, OR strong MACD crossover alone
    Sell: RSI > overbought (58) AND MACD histogram falling, OR strong MACD crossover alone
    """

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.rsi_period    = self.params.get("rsi_period",    14)
        self.rsi_oversold  = self.params.get("rsi_oversold",  42)
        self.rsi_overbought= self.params.get("rsi_overbought",58)
        self.macd_fast     = self.params.get("macd_fast",     12)
        self.macd_slow     = self.params.get("macd_slow",     26)
        self.macd_signal   = self.params.get("macd_signal",   9)
        self.position_pct  = self.params.get("position_pct",  0.1)

    async def analyze(self, candles: list, current_price: float) -> Signal:
        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]
        if len(closes) < self.macd_slow + self.macd_signal + 5:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        rsi_arr = self.rsi(closes, self.rsi_period)
        macd_line, signal_line, histogram = self.macd(
            closes, self.macd_fast, self.macd_slow, self.macd_signal
        )

        curr_rsi  = float(rsi_arr[-1])
        prev_hist = float(histogram[-2])
        curr_hist = float(histogram[-1])
        prev_rsi  = float(rsi_arr[-2])

        if np.isnan(curr_rsi) or np.isnan(curr_hist) or np.isnan(prev_hist):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Indicator NaN")

        # Relative Volume (RVOL) — volume spike confirmation
        vol_arr = np.array(volumes)
        vol_ma  = float(np.mean(vol_arr[-20:])) if len(vol_arr) >= 20 else float(np.mean(vol_arr))
        rvol    = float(vol_arr[-1]) / max(vol_ma, 1e-8)

        hist_rising  = curr_hist > prev_hist
        hist_falling = curr_hist < prev_hist
        macd_cross_up   = prev_hist <= 0 < curr_hist
        macd_cross_down = prev_hist >= 0 > curr_hist

        # ── BUY conditions ──────────────────────────────────────────────
        # 1. Classic: RSI oversold + MACD crossover
        # 2. Relaxed: RSI below threshold + MACD rising + volume spike
        # 3. Strong MACD crossover alone (no RSI filter)
        buy1 = curr_rsi < self.rsi_oversold and macd_cross_up
        buy2 = curr_rsi < self.rsi_oversold and hist_rising and rvol >= 1.3
        buy3 = macd_cross_up and curr_rsi < 55 and rvol >= 1.5

        if buy1 or buy2 or buy3:
            conf = min(1.0, (self.rsi_oversold - curr_rsi) / self.rsi_oversold + 0.3)
            reason_tag = ("RSI+MACD crossover" if buy1
                          else f"RSI oversold+RVOL={rvol:.1f}x" if buy2
                          else f"MACD crossover+RVOL={rvol:.1f}x")
            return Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct,
                reason=f"[RSI+MACD] {reason_tag} RSI={curr_rsi:.1f} hist={curr_hist:.4f}",
                confidence=conf,
                metadata={"rsi": curr_rsi, "macd_hist": curr_hist, "rvol": rvol},
            )

        # ── SELL conditions ─────────────────────────────────────────────
        sell1 = curr_rsi > self.rsi_overbought and macd_cross_down
        sell2 = curr_rsi > self.rsi_overbought and hist_falling and rvol >= 1.3
        sell3 = macd_cross_down and curr_rsi > 45 and rvol >= 1.5

        if sell1 or sell2 or sell3:
            conf = min(1.0, (curr_rsi - self.rsi_overbought) / (100 - self.rsi_overbought) + 0.3)
            reason_tag = ("RSI+MACD crossover" if sell1
                          else f"RSI overbought+RVOL={rvol:.1f}x" if sell2
                          else f"MACD crossover+RVOL={rvol:.1f}x")
            return Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct,
                reason=f"[RSI+MACD] {reason_tag} RSI={curr_rsi:.1f} hist={curr_hist:.4f}",
                confidence=conf,
                metadata={"rsi": curr_rsi, "macd_hist": curr_hist, "rvol": rvol},
            )

        zone = "oversold" if curr_rsi < 45 else "overbought" if curr_rsi > 55 else "neutral"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[RSI+MACD] RSI={curr_rsi:.1f} ({zone}) hist={curr_hist:.5f} RVOL={rvol:.1f}x",
            metadata={"rsi": curr_rsi, "macd_hist": curr_hist, "rvol": rvol},
        )
