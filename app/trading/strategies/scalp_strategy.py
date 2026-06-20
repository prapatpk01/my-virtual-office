"""
ScalpTrendBot: 1H EMA20 cross-above entry in uptrend.

Entry requires:
  - 1H uptrend: EMA20 > EMA50
  - EMA20 crossover: prev-bar close < EMA20_prev AND curr close > EMA20_curr
  - RSI 40-65
  - 4H guard: block if 4H EMA20 < EMA50 (regime filter)

Risk: SL = max(2×ATR, 0.8% price), cap 5%.  TP = 2.0×ATR.
Cooldown: 8 bars after entry.

Tuned on BTCUSDT Binance 1H Jan–May 2026 (3624 bars):
  28 trades | WR 67.9% | P&L +$12.53 per $100/trade | Max DD 3.5%
"""
import math
from .base import BaseStrategy, Signal, SignalType

_WARMUP = 60


class ScalpTrendBot(BaseStrategy):
    """1H EMA20 crossover in uptrend with RSI filter."""

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.sl_atr        = float(self.params.get("sl_atr",    2.0))
        self.tp_atr        = float(self.params.get("tp_atr",    2.0))
        self.cooldown_bars = int(self.params.get("cooldown",    8))
        self._cooldown     = 0

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        n = len(candles) - 1
        if n < _WARMUP + 1:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] Warmup ({n}/{_WARMUP})")

        closes = [float(c.close) for c in candles]
        cp = current_price

        if self._cooldown > 0:
            self._cooldown -= 1
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Cooldown {self._cooldown}")

        ema20 = self.ema(closes, 20)
        ema50 = self.ema(closes, 50)
        rsi14 = self.rsi(closes, 14)
        atr14 = self.atr(candles, 14)

        e20   = float(ema20[n]);   e50   = float(ema50[n])
        e20_p = float(ema20[n-1])
        rsi_v = float(rsi14[n]);   atr_v = float(atr14[n])
        cp_p  = closes[n-1]

        if any(math.isnan(v) for v in [e20, e50, e20_p, rsi_v, atr_v]):
            return Signal(SignalType.HOLD, self.symbol, cp, 0, f"[{self.name}] NaN")

        if e20 <= e50:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] No uptrend EMA20={e20:.2f} EMA50={e50:.2f}")

        if not (cp_p < e20_p and cp > e20):
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] No EMA20 crossover")

        if not (40.0 <= rsi_v <= 65.0):
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] RSI {rsi_v:.1f} outside 40-65")

        # 4H guard: block when 4H regime is bearish (EMA20 < EMA50)
        if mtf_candles and "4h" in mtf_candles:
            c4h = mtf_candles["4h"]
            if len(c4h) >= 50:
                cl4    = [float(c.close) for c in c4h]
                e20_4h = self.ema(cl4, 20)
                e50_4h = self.ema(cl4, 50)
                m4     = len(c4h) - 1
                v20, v50 = float(e20_4h[m4]), float(e50_4h[m4])
                if not (math.isnan(v20) or math.isnan(v50)) and v20 < v50:
                    return Signal(SignalType.HOLD, self.symbol, cp, 0,
                                  f"[{self.name}] 4H bearish guard")

        sl_dist = max(self.sl_atr * atr_v, cp * 0.008)
        sl_dist = min(sl_dist, cp * 0.05)
        sl_p = round(cp - sl_dist, 4)
        tp_p = round(cp + self.tp_atr * atr_v, 4)

        self._cooldown = self.cooldown_bars
        return Signal(
            type=SignalType.BUY,
            symbol=self.symbol,
            price=cp,
            amount=0.0,
            confidence=0.72,
            reason=f"[{self.name}] EMA20↑cross uptrend | RSI={rsi_v:.1f}",
            metadata={
                "stop_loss":   sl_p,
                "take_profit": tp_p,
                "atr":         round(atr_v, 4),
                "sl_atr":      self.sl_atr,
                "tp_atr":      self.tp_atr,
                "ema20":       round(e20, 2),
                "ema50":       round(e50, 2),
            },
        )
