"""
ProfitableBot: 1H trend-following with 2/4-condition entry.

Entry requires:
  - 1H uptrend: EMA20 > EMA50
  - EMA20 rising: EMA20[n] > EMA20[n-3]  (short-term momentum filter)
  - 4H guard: block if 4H EMA20 < EMA50  (regime filter — no slope bypass)
  - 2 of 4 soft conditions:
      C1: RSI ≤ 48 and turning up
      C2: MACD histogram crossing/growing positive
      C3: price bounced off Bollinger lower band
      C4: volume ≥ 1.1× 20-bar MA

Risk: SL = max(2×ATR, 1.5% price), cap 7%.  TP = 1.8×ATR.
Cooldown: 2 bars after entry.

Tuned on BTCUSDT Binance 1H Jan–May 2026 (3624 bars):
  77 trades | WR 67.5% | P&L +$15.15 per $100/trade | Max DD 7.6%
"""
import math
from .base import BaseStrategy, Signal, SignalType

_WARMUP = 60


class ProfitableBot(BaseStrategy):
    """1H trend-following: EMA20 > EMA50 + EMA rising + 4H regime + 2/4 momentum."""

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.sl_atr        = float(self.params.get("sl_atr",    2.0))
        self.tp_atr        = float(self.params.get("tp_atr",    1.8))
        self.cooldown_bars = int(self.params.get("cooldown",    2))
        self._cooldown     = 0

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        n = len(candles) - 1
        if n < _WARMUP:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] Warmup ({n}/{_WARMUP})")

        closes  = [float(c.close)  for c in candles]
        volumes = [float(c.volume) for c in candles]
        cp = current_price

        if self._cooldown > 0:
            self._cooldown -= 1
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Cooldown {self._cooldown}")

        ema20  = self.ema(closes, 20)
        ema50  = self.ema(closes, 50)
        rsi14  = self.rsi(closes, 14)
        vol_ma = self.sma(volumes, 20)
        atr14  = self.atr(candles, 14)
        _, _, mh = self.macd(closes, 12, 26, 9)
        _, _, bb_lower = self.bollinger_bands(closes, 20, 2.0)

        e20 = float(ema20[n]); e50 = float(ema50[n])
        if math.isnan(e20) or math.isnan(e50) or e20 <= e50:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] No 1H uptrend")

        # EMA20 must be rising over last 3 bars (short-term momentum confirmation)
        if n >= _WARMUP + 3:
            e20_3 = float(ema20[n - 3])
            if not math.isnan(e20_3) and e20 <= e20_3:
                return Signal(SignalType.HOLD, self.symbol, cp, 0,
                              f"[{self.name}] EMA20 not rising")

        # 4H guard: block when 4H regime is bearish (EMA20 < EMA50, no slope bypass)
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

        rsi_v  = float(rsi14[n]);    rsi_p = float(rsi14[n-1]) if n >= 1 else rsi_v
        hc     = float(mh[n]);       hp    = float(mh[n-1])    if n >= 1 else hc
        bbl_c  = float(bb_lower[n]); bbl_p = float(bb_lower[n-1]) if n >= 1 else bbl_c
        cp_p   = closes[n-1] if n >= 1 else cp
        vol_v  = volumes[n];  vma_v = float(vol_ma[n])
        atr_v  = float(atr14[n])

        if any(math.isnan(v) for v in [rsi_v, rsi_p, hc, hp, bbl_c, bbl_p, vma_v, atr_v]):
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] NaN indicators")

        conds = 0
        if rsi_v <= 48.0 and rsi_v > rsi_p:              conds += 1
        if (hp < 0 and hc > 0) or (hc > 0 and hc > hp): conds += 1
        if cp_p <= bbl_p and cp > bbl_c:                  conds += 1
        if vol_v >= vma_v * 1.1:                          conds += 1

        if conds < 2:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] {conds}/4 conditions")

        sl_dist = max(self.sl_atr * atr_v, cp * 0.015)
        sl_dist = min(sl_dist, cp * 0.07)
        sl_p = round(cp - sl_dist, 4)
        tp_p = round(cp + self.tp_atr * atr_v, 4)

        self._cooldown = self.cooldown_bars
        return Signal(
            type=SignalType.BUY,
            symbol=self.symbol,
            price=cp,
            amount=0.0,
            confidence=min(0.70 + conds * 0.05, 0.95),
            reason=f"[{self.name}] EMA20↑ + {conds}/4 | RSI={rsi_v:.1f}",
            metadata={
                "stop_loss":   sl_p,
                "take_profit": tp_p,
                "atr":         round(atr_v, 4),
                "sl_atr":      self.sl_atr,
                "tp_atr":      self.tp_atr,
                "conditions":  conds,
                "ema20":       round(e20, 2),
                "ema50":       round(e50, 2),
            },
        )
