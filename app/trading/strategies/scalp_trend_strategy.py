"""
Scalp Trend Strategy — Futures Long & Short
============================================
Trend-continuation scalp adapted from scalp_trend.py, extended to support
both long and short futures positions.

Entry logic (3-layer multi-timeframe):
  BUY  (long)  : 4H up + 1H up + price at EMA20(1H) pullback + 15m bull momentum
  SELL (short) : 4H down + 1H down + price at EMA20(1H) resistance + 15m bear momentum

15m entry requires ALL 4:
  1. close > EMA9  (micro momentum positive / negative)
  2. close > EMA20 (above / below local support)
  3. RSI(14) in 42-65 for longs, 35-58 for shorts (not extreme)
  4. volume ≥ 1.2× MA20

SL/TP: ATR-based, clamped to [sl_min_pct, sl_max_pct].
  LONG : SL = entry − ATR×sl_mult,  TP = entry + ATR×tp_mult
  SHORT: SL = entry + ATR×sl_mult,  TP = entry − ATR×tp_mult

MTF candles required: 1h, 4h (fetched by bot automatically via MTF_TIMEFRAMES).
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class ScalpTrendStrategy(BaseStrategy):

    MTF_TIMEFRAMES = ["1h", "4h"]

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.ema_fast     = self.params.get("ema_fast",     20)
        self.ema_slow     = self.params.get("ema_slow",     50)
        self.ema_micro    = self.params.get("ema_micro",     9)
        self.rsi_period   = self.params.get("rsi_period",   14)
        self.rsi_min_buy  = self.params.get("rsi_min",     42.0)
        self.rsi_max_buy  = self.params.get("rsi_max",     65.0)
        self.rsi_min_sell = self.params.get("rsi_min_sell", 35.0)
        self.rsi_max_sell = self.params.get("rsi_max_sell", 58.0)
        self.pullback_pct = self.params.get("pullback_pct", 0.015)   # ±1.5% of EMA20(1H)
        self.vol_period   = self.params.get("vol_period",   20)
        self.vol_mult     = self.params.get("vol_mult",     1.2)
        self.atr_period   = self.params.get("atr_period",   14)
        self.sl_mult      = self.params.get("sl_mult",      0.8)
        self.tp_mult      = self.params.get("tp_mult",      1.0)     # TP1 for high hit-rate
        self.sl_min_pct   = self.params.get("sl_min_pct",   0.008)
        self.sl_max_pct   = self.params.get("sl_max_pct",   0.04)

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        mtf        = mtf_candles or {}
        candles_1h = mtf.get("1h", [])
        candles_4h = mtf.get("4h", [])

        if len(candles) < 55 or len(candles_1h) < 55 or len(candles_4h) < 55:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[ScalpTrend] Not enough MTF data")

        closes_15m = [c.close for c in candles]
        closes_1h  = [c.close for c in candles_1h]
        closes_4h  = [c.close for c in candles_4h]
        vols_15m   = [c.volume for c in candles]

        # ── Macro trend (4H) ─────────────────────────────────────────────────
        ef4 = self.ema(closes_4h, self.ema_fast)
        es4 = self.ema(closes_4h, self.ema_slow)
        if np.isnan(ef4[-1]) or np.isnan(es4[-1]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[ScalpTrend] 4H EMA not ready")
        macro_up   = bool(ef4[-1] > es4[-1])
        macro_down = bool(ef4[-1] < es4[-1])

        # ── Mid trend (1H) ────────────────────────────────────────────────────
        ef1 = self.ema(closes_1h, self.ema_fast)
        es1 = self.ema(closes_1h, self.ema_slow)
        if np.isnan(ef1[-1]) or np.isnan(es1[-1]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[ScalpTrend] 1H EMA not ready")
        mid_up   = bool(ef1[-1] > es1[-1])
        mid_down = bool(ef1[-1] < es1[-1])

        # ── 1H pullback / resistance zone (use last fully closed 1H bar) ─────
        last_1h   = candles_1h[-2]
        ema20_1h  = float(ef1[-2])
        if ema20_1h == 0:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0)

        near_ema        = abs(last_1h.close - ema20_1h) / ema20_1h <= self.pullback_pct
        wick_bounce_l   = (last_1h.low  <= ema20_1h * 1.003) and (last_1h.close > ema20_1h)
        wick_reject_s   = (last_1h.high >= ema20_1h * 0.997) and (last_1h.close < ema20_1h)
        at_pullback_long  = near_ema or wick_bounce_l
        at_pullback_short = near_ema or wick_reject_s

        # ── 15m entry indicators (last fully closed bar = [-2]) ───────────────
        ef15   = self.ema(closes_15m, self.ema_fast)
        em9    = self.ema(closes_15m, self.ema_micro)
        rsi15  = self.rsi(closes_15m, self.rsi_period)
        volma  = self.sma(vols_15m, self.vol_period)
        atr_a  = self.atr(candles, self.atr_period)

        close_b = closes_15m[-2]
        ema20_b = float(ef15[-2])
        ema9_b  = float(em9[-2])
        rsi_b   = float(rsi15[-2])
        vol_b   = vols_15m[-2]
        volma_b = float(volma[-2])
        atr_v   = float(atr_a[-1])

        if any(np.isnan(v) for v in [ema20_b, ema9_b, rsi_b, volma_b, atr_v]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[ScalpTrend] 15m indicators not ready")

        vol_ok = volma_b > 0 and vol_b >= volma_b * self.vol_mult
        rr = (atr_v * self.tp_mult) / max(atr_v * self.sl_mult, 1e-9)

        def _sl_tp(side: str) -> tuple[float, float]:
            raw  = atr_v * self.sl_mult
            dist = max(current_price * self.sl_min_pct,
                       min(raw, current_price * self.sl_max_pct))
            if side == "long":
                return round(current_price - dist, 2), round(current_price + atr_v * self.tp_mult, 2)
            return round(current_price + dist, 2), round(current_price - atr_v * self.tp_mult, 2)

        # ── BUY (long) ────────────────────────────────────────────────────────
        if macro_up and mid_up and at_pullback_long:
            if (close_b > ema9_b and close_b > ema20_b
                    and self.rsi_min_buy <= rsi_b <= self.rsi_max_buy
                    and vol_ok):
                sl, tp = _sl_tp("long")
                return Signal(
                    SignalType.BUY, self.symbol, current_price,
                    amount=0.08,
                    reason=(f"[ScalpTrend] 4H↑ 1H↑ EMA20-pull "
                            f"RSI={rsi_b:.0f} ATR={atr_v:.0f} RR=1:{rr:.1f}"),
                    confidence=0.70,
                    metadata={"stop_loss": sl, "take_profit": tp, "atr": atr_v},
                )

        # ── SELL (short) ──────────────────────────────────────────────────────
        if macro_down and mid_down and at_pullback_short:
            if (close_b < ema9_b and close_b < ema20_b
                    and self.rsi_min_sell <= rsi_b <= self.rsi_max_sell
                    and vol_ok):
                sl, tp = _sl_tp("short")
                return Signal(
                    SignalType.SELL, self.symbol, current_price,
                    amount=0.08,
                    reason=(f"[ScalpTrend] 4H↓ 1H↓ EMA20-resist "
                            f"RSI={rsi_b:.0f} ATR={atr_v:.0f} RR=1:{rr:.1f}"),
                    confidence=0.70,
                    metadata={"stop_loss": sl, "take_profit": tp, "atr": atr_v},
                )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            reason=(f"[ScalpTrend] "
                    f"4H={'↑' if macro_up else '↓' if macro_down else '→'} "
                    f"1H={'↑' if mid_up else '↓' if mid_down else '→'} "
                    f"pull-L={at_pullback_long} pull-S={at_pullback_short}"),
        )
