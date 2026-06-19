"""
Smart Money Strategy — Futures Long & Short
============================================
Port of smart_money style from unified_trading_bot.py.

Uses market structure (BOS/CHoCH) + EMA alignment + multi-TF trend agreement.

All 4 gates must pass to trigger:
  - composite confidence  ≥ min_confidence  (75)
  - multi-TF score        ≥ min_multi_tf    (70)
  - BOS/CHoCH score       ≥ min_bos_choch   (60)
  - EMA cross score       ≥ min_ema_cross   (60)
  - all 3 components must agree on direction

SL/TP: ATR-based
  sl_mult=1.8, rr=1.8 → R:R 1:1.8  (break-even WR 35.7%)

⚠️ BOS/CHoCH scoring is a simplified fractal implementation — validate with
   backtest before relying on it in live trading.
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class SmartMoneyStrategy(BaseStrategy):

    MTF_TIMEFRAMES = ["1h", "4h"]

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.ema_fast       = self.params.get("ema_fast",        9)
        self.ema_slow       = self.params.get("ema_slow",       21)
        self.swing_lookback = self.params.get("swing_lookback",   5)
        self.atr_period     = self.params.get("atr_period",     14)
        self.sl_mult        = self.params.get("sl_mult",        1.8)
        self.rr             = self.params.get("rr",             1.8)
        self.min_confidence = self.params.get("min_confidence", 42.0)  # ↓ from 55
        self.min_multi_tf   = self.params.get("min_multi_tf",   40.0)  # ↓ from 55
        self.min_bos_choch  = self.params.get("min_bos_choch",  35.0)  # ↓ from 50
        self.min_ema_cross  = self.params.get("min_ema_cross",  35.0)  # ↓ from 50

    # ── Market structure helpers ──────────────────────────────────────────

    def _swing_points(self, candles: list) -> list:
        lb = self.swing_lookback
        n  = len(candles)
        highs = np.array([c.high for c in candles])
        lows  = np.array([c.low  for c in candles])
        swings = []
        for i in range(lb, n - lb):
            if highs[i] == highs[i - lb: i + lb + 1].max():
                swings.append((i, float(highs[i]), "high"))
            if lows[i] == lows[i - lb: i + lb + 1].min():
                swings.append((i, float(lows[i]), "low"))
        return swings

    def _analyze_structure(self, candles: list) -> dict:
        swings = self._swing_points(candles)
        if len(swings) < 4:
            return {"bias": "neutral", "score": 0.0}

        sh = [(i, v) for i, v, t in swings if t == "high"]
        sl = [(i, v) for i, v, t in swings if t == "low"]
        if len(sh) < 2 or len(sl) < 2:
            return {"bias": "neutral", "score": 0.0}

        last_h, prev_h = sh[-1][1], sh[-2][1]
        last_l, prev_l = sl[-1][1], sl[-2][1]
        making_hh_hl = (last_h > prev_h) and (last_l > prev_l)
        making_lh_ll = (last_h < prev_h) and (last_l < prev_l)

        cur_close = float(candles[-2].close)
        if cur_close > last_h:
            strength = min((cur_close - last_h) / last_h * 100, 2.0) / 2.0
            score = 60 + strength * 40 if making_hh_hl else 55 + strength * 35
            return {"bias": "bullish", "score": score}
        if cur_close < last_l:
            strength = min((last_l - cur_close) / last_l * 100, 2.0) / 2.0
            score = 60 + strength * 40 if making_lh_ll else 55 + strength * 35
            return {"bias": "bearish", "score": score}
        return {"bias": "neutral", "score": 0.0}

    def _ema_cross_score(self, closes: list) -> dict:
        ef = self.ema(closes, self.ema_fast)
        es = self.ema(closes, self.ema_slow)
        if len(ef) < 8 or np.isnan(ef[-2]) or np.isnan(es[-2]) or es[-2] == 0:
            return {"bias": "neutral", "score": 0.0}
        ef_cur, es_cur = float(ef[-2]), float(es[-2])
        ef_prev = float(ef[-7]) if not np.isnan(ef[-7]) else ef_cur
        dist_pct = abs(ef_cur - es_cur) / es_cur * 100
        bias = "bullish" if ef_cur > es_cur else "bearish"
        slope = (ef_cur - ef_prev) / ef_prev * 100 if ef_prev else 0
        slope_aligned = (slope > 0 and bias == "bullish") or (slope < 0 and bias == "bearish")
        score = 50 + min(dist_pct * 20, 30) + (20 if slope_aligned else 0)
        return {"bias": bias, "score": min(score, 100.0)}

    def _tf_trend(self, closes: list) -> str:
        if len(closes) < self.ema_slow + 2:
            return "flat"
        ef = self.ema(closes, self.ema_fast)
        es = self.ema(closes, self.ema_slow)
        if np.isnan(ef[-2]) or np.isnan(es[-2]):
            return "flat"
        return "up" if float(ef[-2]) > float(es[-2]) else "down"

    def _multi_tf_score(self, closes_15m: list, closes_1h: list, closes_4h: list) -> dict:
        trends = [self._tf_trend(c) for c in (closes_15m, closes_1h, closes_4h)]
        up, dn = trends.count("up"), trends.count("down")
        if up == 3: return {"bias": "bullish", "score": 100.0}
        if dn == 3: return {"bias": "bearish", "score": 100.0}
        if up == 2: return {"bias": "bullish", "score": 65.0}
        if dn == 2: return {"bias": "bearish", "score": 65.0}
        return {"bias": "neutral", "score": 30.0}

    def _composite(self, mtf: dict, ema_c: dict, struct: dict) -> tuple[float, str]:
        # Majority vote: 2/3 components agreeing on direction is enough
        biases = [mtf["bias"], ema_c["bias"], struct["bias"]]
        bull = biases.count("bullish")
        bear = biases.count("bearish")
        if bull >= 2:
            direction, agree = "bullish", 1.0 if bull == 3 else 0.8
        elif bear >= 2:
            direction, agree = "bearish", 1.0 if bear == 3 else 0.8
        else:
            direction, agree = "neutral", 0.5
        raw = mtf["score"] * 0.35 + ema_c["score"] * 0.25 + struct["score"] * 0.40
        return raw * agree, direction

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        mtf        = mtf_candles or {}
        candles_1h = mtf.get("1h", [])
        candles_4h = mtf.get("4h", [])

        min_len = self.swing_lookback * 2 + 10
        if len(candles) < min_len or len(candles_1h) < 55 or len(candles_4h) < 55:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[SmartMoney] Not enough data")

        closes_15m = [c.close for c in candles]
        closes_1h  = [c.close for c in candles_1h]
        closes_4h  = [c.close for c in candles_4h]

        mtf_score  = self._multi_tf_score(closes_15m, closes_1h, closes_4h)
        ema_score  = self._ema_cross_score(closes_15m)
        struct     = self._analyze_structure(candles)
        confidence, direction_bias = self._composite(mtf_score, ema_score, struct)

        if not (confidence >= self.min_confidence and
                mtf_score["score"] >= self.min_multi_tf and
                struct["score"]   >= self.min_bos_choch and
                ema_score["score"] >= self.min_ema_cross and
                direction_bias in ("bullish", "bearish")):
            return Signal(
                SignalType.HOLD, self.symbol, current_price, 0,
                reason=(f"[SmartMoney] conf={confidence:.0f} "
                        f"mtf={mtf_score['score']:.0f} bos={struct['score']:.0f} "
                        f"ema={ema_score['score']:.0f} bias={direction_bias}"),
            )

        atr_a = self.atr(candles, self.atr_period)
        atr_v = float(atr_a[-1])
        if np.isnan(atr_v) or atr_v <= 0:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[SmartMoney] ATR not ready")

        direction = "long" if direction_bias == "bullish" else "short"
        sl_dist   = atr_v * self.sl_mult
        tp_dist   = sl_dist * self.rr
        if direction == "long":
            sl, tp   = round(current_price - sl_dist, 2), round(current_price + tp_dist, 2)
            sig_type = SignalType.BUY
        else:
            sl, tp   = round(current_price + sl_dist, 2), round(current_price - tp_dist, 2)
            sig_type = SignalType.SELL

        return Signal(
            sig_type, self.symbol, current_price,
            amount=0.08,
            reason=(f"[SmartMoney] {direction.upper()} conf={confidence:.0f} "
                    f"mtf={mtf_score['score']:.0f} bos={struct['score']:.0f} "
                    f"ema={ema_score['score']:.0f} RR=1:{self.rr:.1f}"),
            confidence=min(confidence / 100.0, 0.90),
            metadata={"stop_loss": sl, "take_profit": tp, "atr": atr_v},
        )
