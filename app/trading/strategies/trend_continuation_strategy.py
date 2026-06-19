"""
Trend Continuation Strategy — Futures Long & Short
===================================================
Port of trend_continuation style from unified_trading_bot.py.

Entry logic (3-layer MTF):
  BUY  (long)  : 4H up + 1H up + 15m at EMA20 pullback + ALL 4 conditions
  SELL (short) : 4H down + 1H down + 15m at EMA20 resistance + ALL 4 conditions

15m entry requires ALL 4:
  1. close > EMA9  (micro momentum)
  2. close > EMA20 (above local support)
  3. RSI in [rsi_min, rsi_max]
  4. volume ≥ vol_mult × MA20

SL/TP: ATR-based, clamped to [sl_min_pct, sl_max_pct]
  sl_mult=0.8, tp_mult=2.0 → R:R 1:2.5  (break-even WR 28.6%)
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class TrendContinuationStrategy(BaseStrategy):

    MTF_TIMEFRAMES = ["1h", "4h"]

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.ema_fast     = self.params.get("ema_fast",      20)
        self.ema_slow     = self.params.get("ema_slow",      50)
        self.ema_micro    = self.params.get("ema_micro",      9)
        self.rsi_period   = self.params.get("rsi_period",    14)
        self.rsi_min_buy  = self.params.get("rsi_min",      35.0)  # ↓ from 38
        self.rsi_max_buy  = self.params.get("rsi_max",      75.0)  # ↑ from 72
        self.bias_gate    = self.params.get("bias_gate",    15.0)  # comp_pct threshold
        self.rsi_min_sell = self.params.get("rsi_min_sell", 28.0)  # ↓ from 35
        self.rsi_max_sell = self.params.get("rsi_max_sell", 65.0)  # ↑ from 58
        self.pullback_pct = self.params.get("pullback_pct", 0.025) # ↑ from 0.015 ±2.5%
        self.vol_period   = self.params.get("vol_period",    20)
        self.vol_mult     = self.params.get("vol_mult",      1.0)  # ↓ from 1.2
        self.min_entry_cond = self.params.get("min_entry_cond", 3) # 3/4 not ALL 4
        self.atr_period   = self.params.get("atr_period",   14)
        self.sl_mult      = self.params.get("sl_mult",       0.8)
        self.tp_mult      = self.params.get("tp_mult",       4.0)  # ↑ R:R 1:5 → BE WR 16.7%
        self.sl_min_pct   = self.params.get("sl_min_pct",  0.002)
        self.sl_max_pct   = self.params.get("sl_max_pct",  0.020)  # ↑ from 0.012

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        mtf        = mtf_candles or {}
        candles_1h = mtf.get("1h", [])
        candles_4h = mtf.get("4h", [])

        if len(candles) < 55 or len(candles_1h) < 55 or len(candles_4h) < 55:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[TrendCont] Not enough MTF data")

        closes_15m = [c.close for c in candles]
        closes_1h  = [c.close for c in candles_1h]
        closes_4h  = [c.close for c in candles_4h]
        vols_15m   = [c.volume for c in candles]

        # ── Macro trend (4H): EMA20 vs EMA50 ─────────────────────────────
        ef4 = self.ema(closes_4h, self.ema_fast)
        es4 = self.ema(closes_4h, self.ema_slow)
        if np.isnan(ef4[-1]) or np.isnan(es4[-1]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[TrendCont] 4H EMA not ready")
        macro_up   = bool(ef4[-1] > es4[-1])
        macro_down = bool(ef4[-1] < es4[-1])

        # ── Mid trend (1H): EMA20 vs EMA50 ───────────────────────────────
        ef1 = self.ema(closes_1h, self.ema_fast)
        es1 = self.ema(closes_1h, self.ema_slow)
        if np.isnan(ef1[-1]) or np.isnan(es1[-1]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[TrendCont] 1H EMA not ready")
        mid_up   = bool(ef1[-1] > es1[-1])
        mid_down = bool(ef1[-1] < es1[-1])

        # ── 1H pullback / resistance zone (last closed 1H bar = [-2]) ────
        last_1h  = candles_1h[-2]
        ema20_1h = float(ef1[-2])
        if ema20_1h == 0 or np.isnan(ema20_1h):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0)

        near_ema          = abs(last_1h.close - ema20_1h) / ema20_1h <= self.pullback_pct
        wick_bounce_l     = (last_1h.low  <= ema20_1h * 1.003) and (last_1h.close > ema20_1h)
        wick_reject_s     = (last_1h.high >= ema20_1h * 0.997) and (last_1h.close < ema20_1h)
        at_pullback_long  = near_ema or wick_bounce_l
        at_pullback_short = near_ema or wick_reject_s

        # ── MTF bias pre-filter: 15m+1H+4H alignment gate ────────────────
        comp_pct, bias_label = self.compute_mtf_bias(
            candles, mtf_candles,
            ema_fast=20, ema_slow=50, rsi_period=14, rsi_bull=55.0, rsi_bear=45.0,
        )
        long_bias_ok  = comp_pct >  self.bias_gate
        short_bias_ok = comp_pct < -self.bias_gate
        if not (long_bias_ok or short_bias_ok):
            return Signal(
                SignalType.HOLD, self.symbol, current_price, 0,
                reason=(f"[TrendCont] MTF bias too weak: comp={comp_pct:.0f} ({bias_label}) "
                        f"need >{self.bias_gate:.0f} for long / <-{self.bias_gate:.0f} for short"),
            )

        # ── 15m indicators (last closed bar = [-2]) ───────────────────────
        ef15  = self.ema(closes_15m, self.ema_fast)
        em9   = self.ema(closes_15m, self.ema_micro)
        rsi15 = self.rsi(closes_15m, self.rsi_period)
        volma = self.sma(vols_15m, self.vol_period)
        atr_a = self.atr(candles, self.atr_period)

        close_b = closes_15m[-2]
        ema20_b = float(ef15[-2])
        ema9_b  = float(em9[-2])
        rsi_b   = float(rsi15[-2])
        vol_b   = vols_15m[-2]
        volma_b = float(volma[-2])
        atr_v   = float(atr_a[-1])

        if any(np.isnan(v) for v in [ema20_b, ema9_b, rsi_b, atr_v]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[TrendCont] 15m indicators not ready")

        vol_ok = volma_b > 0 and vol_b >= volma_b * self.vol_mult
        rr     = self.tp_mult / max(self.sl_mult, 1e-9)

        def _sl_tp(side: str) -> tuple[float, float]:
            raw  = atr_v * self.sl_mult
            dist = max(current_price * self.sl_min_pct,
                       min(raw, current_price * self.sl_max_pct))
            tp_d = atr_v * self.tp_mult
            if side == "long":
                return round(current_price - dist, 2), round(current_price + tp_d, 2)
            return round(current_price + dist, 2), round(current_price - tp_d, 2)

        # ── BUY: 4H up + 1H up + pullback + bias aligned + min_entry_cond/4 ─
        if macro_up and mid_up and at_pullback_long and long_bias_ok:
            c1 = close_b > ema9_b
            c2 = close_b > ema20_b
            c3 = self.rsi_min_buy <= rsi_b <= self.rsi_max_buy
            c4 = vol_ok
            met = sum([c1, c2, c3, c4])
            if met >= self.min_entry_cond:
                sl, tp = _sl_tp("long")
                return Signal(
                    SignalType.BUY, self.symbol, current_price,
                    amount=0.08,
                    reason=(f"[TrendCont] 4H↑ 1H↑ bias={comp_pct:.0f} EMA20pull "
                            f"cond={met}/4 RSI={rsi_b:.0f} ATR={atr_v:.0f} RR=1:{rr:.2f}"),
                    confidence=0.65 + met * 0.02,
                    metadata={"stop_loss": sl, "take_profit": tp, "atr": atr_v},
                )

        # ── SELL: 4H down + 1H down + resistance + bias aligned + min_entry_cond/4
        if macro_down and mid_down and at_pullback_short and short_bias_ok:
            c1 = close_b < ema9_b
            c2 = close_b < ema20_b
            c3 = self.rsi_min_sell <= rsi_b <= self.rsi_max_sell
            c4 = vol_ok
            met = sum([c1, c2, c3, c4])
            if met >= self.min_entry_cond:
                sl, tp = _sl_tp("short")
                return Signal(
                    SignalType.SELL, self.symbol, current_price,
                    amount=0.08,
                    reason=(f"[TrendCont] 4H↓ 1H↓ bias={comp_pct:.0f} EMA20resist "
                            f"cond={met}/4 RSI={rsi_b:.0f} ATR={atr_v:.0f} RR=1:{rr:.2f}"),
                    confidence=0.65 + met * 0.02,
                    metadata={"stop_loss": sl, "take_profit": tp, "atr": atr_v},
                )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            reason=(f"[TrendCont] "
                    f"4H={'↑' if macro_up else '↓' if macro_down else '→'} "
                    f"1H={'↑' if mid_up else '↓' if mid_down else '→'} "
                    f"bias={comp_pct:.0f} pull-L={at_pullback_long} pull-S={at_pullback_short}"),
        )
