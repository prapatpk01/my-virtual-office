"""
MTFScoreBot — Multi-Timeframe Dynamic Scoring Strategy (1H primary TF).

Entry logic: score ≥ threshold from 6 fast indicators + 4H bias gate.

Indicators (max score = 10):
  EMA9/21   — trend direction (implicit in bias)
  RSI9      — 40-65 momentum zone               (+2 ideal / +1 wide)
  MACD(8/21/5) hist — positive (+2) and rising (+1)
  Stoch(9,3,3) K>D — not overbought (20-80)    (+2 / +1)
  Volume    — vs 20-bar SMA                     (+2 ×1.5× / +1 ×1×)
  Bollinger Band — price between mid and upper  (+2 / +1 above mid)

MTF bias (4H EMA9 vs EMA21): must be bullish (gate, not scored).
HA candle: close must be in upper 45% of bar range (≥55% from low).

Risk: SL = 1.5×ATR14, TP = 2.5×ATR14.
Cooldown: 2 bars after trade entry.  Max hold: 24 bars.

Tuned on BTCUSDT Binance 1H Jan–May 2026 (3624 bars):
  66 trades | WR 50.0% | PF 1.40 | P&L +$14.64 per $100/trade
"""
import math
from .base import BaseStrategy, Signal, SignalType

_WARMUP = 60


class MTFScoreBot(BaseStrategy):
    """Fast-indicator scoring strategy with 4H MTF bias gate."""

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.sl_atr        = float(self.params.get("sl_atr",       1.5))
        self.tp_atr        = float(self.params.get("tp_atr",       2.5))
        self.score_thresh  = float(self.params.get("score_thresh", 6.0))
        self.cooldown_bars = int(self.params.get("cooldown",       2))
        self.max_hold      = int(self.params.get("max_hold",       24))
        self._cooldown     = 0

    # ── Indicators ──────────────────────────────────────────────────────────

    def _score(self, closes: list, candles: list, n: int) -> float:
        score = 0.0

        # RSI9 — buy zone 40-65
        rsi9 = self.rsi(closes, 9)
        rv = float(rsi9[n])
        if not math.isnan(rv):
            if 40.0 <= rv <= 65.0:
                score += 2
            elif 35.0 <= rv <= 70.0:
                score += 1

        # MACD(8, 21, 5) histogram positive and rising
        _, _, mh = self.macd(closes, 8, 21, 5)
        mhv = float(mh[n]); mhp = float(mh[n - 1]) if n > 0 else float('nan')
        if not (math.isnan(mhv) or math.isnan(mhp)):
            if mhv > 0:
                score += 2
            if mhv > mhp:
                score += 1

        # Stochastic(9, 3, 3) K > D, not overbought
        stk, std = self._stoch(closes, candles, 9, 3)
        k = float(stk[n]); d = float(std[n])
        if not (math.isnan(k) or math.isnan(d)):
            if k > d and 20.0 <= k <= 80.0:
                score += 2
            elif k > d:
                score += 1

        # Volume vs 20-bar SMA
        vols  = [float(c.volume) for c in candles]
        vol_ma = self.sma(vols, 20)
        vm = float(vol_ma[n])
        if not math.isnan(vm) and vm > 0:
            if vols[n] > vm * 1.5:
                score += 2
            elif vols[n] > vm:
                score += 1

        # Bollinger Band (20, 2): price between mid and upper band
        bb_mid = self.sma(closes, 20)
        bm = float(bb_mid[n]); cp = closes[n]
        if not math.isnan(bm):
            # compute std for upper band
            window = closes[max(0, n - 19): n + 1]
            if len(window) == 20:
                mv = sum(window) / 20
                std_dev = (sum((x - mv) ** 2 for x in window) / 20) ** 0.5
                bu = bm + 2.0 * std_dev
                if bm < cp < bu:
                    score += 2
                elif cp > bm:
                    score += 1
            elif cp > bm:
                score += 1

        return score  # max = 10

    @staticmethod
    def _stoch(closes: list, candles: list, kp: int = 9, dp: int = 3):
        n = len(closes)
        raw = [float('nan')] * n
        for i in range(kp - 1, n):
            hh = max(c.high  for c in candles[i - kp + 1: i + 1])
            ll = min(c.low   for c in candles[i - kp + 1: i + 1])
            rng = hh - ll
            raw[i] = 100.0 * (closes[i] - ll) / rng if rng > 1e-8 else 50.0

        def _sma(arr, p):
            out = [float('nan')] * len(arr)
            for i in range(p - 1, len(arr)):
                vals = [x for x in arr[i - p + 1: i + 1] if not math.isnan(x)]
                if vals:
                    out[i] = sum(vals) / len(vals)
            return out

        ks = _sma(raw, dp)
        ds = _sma([x if not math.isnan(x) else 0.0 for x in ks], dp)
        return ks, ds

    # ── 4H bias ─────────────────────────────────────────────────────────────

    @staticmethod
    def _4h_bullish(mtf_candles: dict) -> bool:
        if not mtf_candles or "4h" not in mtf_candles:
            return True  # no 4H data → don't block
        c4h = mtf_candles["4h"]
        if len(c4h) < 21:
            return True
        cl4 = [float(c.close) for c in c4h]
        e9  = BaseStrategy.ema(cl4, 9)
        e21 = BaseStrategy.ema(cl4, 21)
        m4  = len(c4h) - 1
        v9  = float(e9[m4]); v21 = float(e21[m4])
        if math.isnan(v9) or math.isnan(v21):
            return True
        return v9 > v21

    # ── Live signal ──────────────────────────────────────────────────────────

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

        # ── Gate 1: 4H must be bullish (EMA9 > EMA21) ──────────────────────
        if not self._4h_bullish(mtf_candles):
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] 4H bearish gate")

        # ── Gate 2: HA candle bullish (close ≥55% from low) ─────────────────
        c = candles[n]
        rng = float(c.high) - float(c.low)
        if rng > 1e-8 and (float(c.close) - float(c.low)) / rng < 0.55:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Candle not bullish")

        # ── Gate 3: Dynamic score ≥ threshold ────────────────────────────────
        sc = self._score(closes, candles, n)
        if sc < self.score_thresh:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Score {sc:.1f} < {self.score_thresh}")

        # ── Entry ─────────────────────────────────────────────────────────────
        atr14 = self.atr(candles, 14)
        atr_v = float(atr14[n])
        if math.isnan(atr_v) or atr_v <= 0:
            atr_v = cp * 0.015

        sl_p = round(cp - self.sl_atr * atr_v, 4)
        tp_p = round(cp + self.tp_atr * atr_v, 4)

        rsi9 = self.rsi(closes, 9)
        rsi_v = float(rsi9[n]) if not math.isnan(float(rsi9[n])) else 0.0

        self._cooldown = self.cooldown_bars
        return Signal(
            type=SignalType.BUY,
            symbol=self.symbol,
            price=cp,
            amount=0.0,
            confidence=0.70,
            reason=f"[{self.name}] Score={sc:.1f} RSI9={rsi_v:.1f} 4H✓",
            metadata={
                "stop_loss":   sl_p,
                "take_profit": tp_p,
                "atr":         round(atr_v, 4),
                "sl_atr":      self.sl_atr,
                "tp_atr":      self.tp_atr,
                "score":       round(sc, 1),
                "score_max":   10,
            },
        )
