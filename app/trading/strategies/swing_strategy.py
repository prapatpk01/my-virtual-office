"""
Swing Strategies v5 — Wide Config (live trading).

SwingReversalStrategy: MACD-cross timing in soft-bull regime (EMA80 > EMA200 × 0.98).
  RSI 30-65 | SL=2.0×ATR | TP=2.0×ATR | hold≤3d | ADX(14) > 15

Entry filter:
  soft-bull  : EMA80 >= EMA200 × 0.98
  ADX(14)    : > 15 (trend has momentum — skip sideways)
  candle     : close > 60% of bar range
  MACD cross : histogram crossed neg→pos within last 3 bars
  RSI        : in [rsi_lo, rsi_hi]
  Volume     : >= 20-bar SMA × vol_mult

Tuned on BTCUSDT Binance 1H Jan–May 2026 (3624 bars):
  35 trades | WR 62.9% | PF 1.75 | P&L +$12.23 per $100/trade
"""
import math
import numpy as np
from .base import BaseStrategy, Signal, SignalType

_WARMUP = 210   # EMA200 (200) + MACD warmup (26+9) — need at least this many bars


class SwingStrategy(BaseStrategy):
    """Base swing strategy — parameterised. Use the named subclasses directly."""

    _DEFAULTS: dict = {}

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        d = self._DEFAULTS
        self.sl_atr    = float(self.params.get("sl_atr",        d.get("sl_atr",        2.5)))
        self.tp_atr    = float(self.params.get("tp_atr",        d.get("tp_atr",        1.5)))
        self.rsi_lo    = float(self.params.get("rsi_lo",        d.get("rsi_lo",       34.0)))
        self.rsi_hi    = float(self.params.get("rsi_hi",        d.get("rsi_hi",       62.0)))
        self.vol_mult  = float(self.params.get("vol_mult",      d.get("vol_mult",      1.2)))
        self.adx_thresh= float(self.params.get("adx_thresh",   d.get("adx_thresh",   15.0)))
        self.max_hold  = int(float(self.params.get("max_hold_days", d.get("max_hold_days", 3))) * 24)
        self._cooldown = 0

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _macd_cross_up(hist: np.ndarray, i: int, lookback: int = 3) -> bool:
        for j in range(i, max(i - lookback, 0), -1):
            if j < 1:
                continue
            hj, hj1 = float(hist[j]), float(hist[j - 1])
            if math.isnan(hj) or math.isnan(hj1):
                continue
            if hj1 < 0 and hj > 0:
                return True
        return False

    @staticmethod
    def _candle_bullish(c) -> bool:
        r = float(c.high) - float(c.low)
        return r > 1e-6 and (float(c.close) - float(c.low)) / r >= 0.60

    def _entry_ok(self, candles, closes, ema80, ema200,
                  rsi14, macd_hist, vol_ma, i: int) -> bool:
        e80 = float(ema80[i]);  e200 = float(ema200[i])
        if math.isnan(e80) or math.isnan(e200):
            return False
        if e80 < e200 * 0.98:          # soft-bull regime
            return False
        if not self._candle_bullish(candles[i]):
            return False
        rsi_v = float(rsi14[i])  if not math.isnan(float(rsi14[i]))  else 50.0
        vol_v = float(candles[i].volume)
        vma_v = float(vol_ma[i]) if not math.isnan(float(vol_ma[i])) else 1.0
        return (
            self._macd_cross_up(macd_hist, i) and
            self.rsi_lo <= rsi_v <= self.rsi_hi and
            vol_v >= vma_v * self.vol_mult
        )

    # ── Live analysis ──────────────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        if len(candles) < _WARMUP:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] Warming up ({len(candles)}/{_WARMUP})")

        closes  = np.array([c.close  for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        ema80    = self.ema(closes.tolist(), 80)
        ema200   = self.ema(closes.tolist(), 200)
        rsi14    = self.rsi(closes.tolist(), 14)
        vol_ma   = self.sma(volumes.tolist(), 20)
        atr14    = self.atr(candles, 14)
        _, _, mh = self.macd(closes.tolist(), 12, 26, 9)

        n = len(candles) - 1

        if self._cooldown > 0:
            self._cooldown -= 1
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] Cooldown {self._cooldown} bars left")

        # ADX gate — skip entries when trend has no momentum (sideway market)
        adx_arr, _, _ = self.adx(candles, 14)
        adx_v = float(adx_arr[n])
        if not math.isnan(adx_v) and adx_v < self.adx_thresh:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] ADX={adx_v:.1f} < {self.adx_thresh}")

        if self._entry_ok(candles, closes, ema80, ema200, rsi14, mh, vol_ma, n):
            atr_v = (float(atr14[n]) if not math.isnan(float(atr14[n]))
                     else current_price * 0.015)
            sl_p  = round(current_price - self.sl_atr * atr_v, 4)
            tp_p  = round(current_price + self.tp_atr * atr_v, 4)
            rsi_v = float(rsi14[n]) if not math.isnan(float(rsi14[n])) else 0.0
            # Self-throttle: block re-entry for the full expected hold duration.
            # max_hold is already in bars (hours on 1H TF).
            self._cooldown = self.max_hold
            return Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                price=current_price,
                amount=0.0,
                confidence=0.75,
                reason=f"[{self.name}] MACD↑ RSI={rsi_v:.1f} soft-bull",
                metadata={
                    "stop_loss":   sl_p,
                    "take_profit": tp_p,
                    "atr":         round(atr_v, 4),
                    "sl_atr":      self.sl_atr,
                    "tp_atr":      self.tp_atr,
                    "ema80":       round(float(ema80[n]),  2),
                    "ema200":      round(float(ema200[n]), 2),
                },
            )

        e80  = float(ema80[n])  if not math.isnan(float(ema80[n]))  else 0.0
        e200 = float(ema200[n]) if not math.isnan(float(ema200[n])) else 0.0
        rsi_v = float(rsi14[n]) if not math.isnan(float(rsi14[n])) else 0.0
        trend = "soft-bull" if e80 >= e200 * 0.98 else "bear"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[{self.name}] {trend} | RSI={rsi_v:.1f}",
            metadata={"trend": trend, "ema80": round(e80, 2), "ema200": round(e200, 2)},
        )


# ── Named variants ─────────────────────────────────────────────────────────────

class SwingReversalStrategy(SwingStrategy):
    _DEFAULTS = dict(sl_atr=2.0, tp_atr=2.0, rsi_lo=30.0, rsi_hi=65.0,
                     vol_mult=1.0, adx_thresh=15.0, max_hold_days=3)
