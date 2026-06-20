"""
Trend Continuation Strategy — Futures Long & Short  (v2 — earlier entries / TP1+TP2)
==================================================================================
Buy-the-dip / sell-the-rip in the direction of the higher-TF trend.

Entry logic (3-layer MTF):
  BUY  (long)  : 4H up + 1H up + Supertrend(4H)=green + 1H pullback to EMA20±0.5ATR
                 + MTF bias > bias_gate + min_entry_cond/4
  SELL (short) : 4H down + 1H down + Supertrend(4H)=red + 1H rally to EMA20±0.5ATR
                 + MTF bias < -bias_gate + min_entry_cond/4

Changes vs v1:
  • bias_gate 70 → 55  (catch the trend earlier, fewer missed legs)
  • pullback zone: dynamic EMA20 ± pullback_atr×ATR(1H)  (was fixed 1.0%)
  • NEW Supertrend(4H) confirmation filter — direction must match the order

15m entry requires min_entry_cond/4:
  1. close > EMA9   2. close > EMA20   3. RSI in band   4. volume ≥ vol_mult×MA20

Risk (Global): Initial SL 1.5×ATR, TP1 1.2×ATR (close 50% → SL→BE), TP2 2.2×ATR.
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
        self.rsi_min_buy  = self.params.get("rsi_min",      35.0)
        self.rsi_max_buy  = self.params.get("rsi_max",      75.0)
        self.bias_gate    = self.params.get("bias_gate",    55.0)  # ↓ from 70 — earlier entries
        self.rsi_min_sell = self.params.get("rsi_min_sell", 28.0)
        self.rsi_max_sell = self.params.get("rsi_max_sell", 65.0)
        self.pullback_atr = self.params.get("pullback_atr",  0.5)  # NEW: EMA20 ± 0.5×ATR(1H) zone
        self.vol_period   = self.params.get("vol_period",    20)
        self.vol_mult     = self.params.get("vol_mult",      1.0)
        self.min_entry_cond = self.params.get("min_entry_cond", 4)
        self.atr_period   = self.params.get("atr_period",   14)
        self.st_period    = self.params.get("st_period",    10)   # NEW: Supertrend(4H)
        self.st_mult      = self.params.get("st_mult",       3.0)
        # Global-Risk multiples (grid-tuned: SL 1.2 / TP1 1.5 / TP2 3.0 ATR)
        self.sl_atr       = self.params.get("sl_atr",        1.2)
        self.tp1_atr      = self.params.get("tp1_atr",       1.5)
        self.tp2_atr      = self.params.get("tp2_atr",       3.0)
        self.sl_min_pct   = self.params.get("sl_min_pct",  0.010)
        self.sl_max_pct   = self.params.get("sl_max_pct",  0.035)
        self.risk_pct     = self.params.get("risk_pct",     0.02)
        self.partial_pct  = self.params.get("partial_pct",   0.5)

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

        # ── Supertrend(4H) confirmation (NEW) ─────────────────────────────
        _, st_dir = self.supertrend(candles_4h, self.st_period, self.st_mult)
        st_now    = int(st_dir[-2]) if len(st_dir) >= 2 else 0   # last closed 4H bar
        st_green  = st_now > 0
        st_red    = st_now < 0

        # ── Mid trend (1H): EMA20 vs EMA50 ───────────────────────────────
        ef1 = self.ema(closes_1h, self.ema_fast)
        es1 = self.ema(closes_1h, self.ema_slow)
        if np.isnan(ef1[-1]) or np.isnan(es1[-1]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[TrendCont] 1H EMA not ready")
        mid_up   = bool(ef1[-1] > es1[-1])
        mid_down = bool(ef1[-1] < es1[-1])

        # ── 1H pullback zone: EMA20 ± pullback_atr×ATR(1H) (last closed = [-2]) ──
        atr1h_a  = self.atr(candles_1h, self.atr_period)
        atr1h    = float(atr1h_a[-2]) if not np.isnan(atr1h_a[-2]) else 0.0
        last_1h  = candles_1h[-2]
        ema20_1h = float(ef1[-2])
        if ema20_1h == 0 or np.isnan(ema20_1h) or atr1h <= 0:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[TrendCont] 1H pullback not ready")
        zone = self.pullback_atr * atr1h
        near_ema          = abs(last_1h.close - ema20_1h) <= zone
        wick_bounce_l     = (last_1h.low  <= ema20_1h + zone) and (last_1h.close > ema20_1h)
        wick_reject_s     = (last_1h.high >= ema20_1h - zone) and (last_1h.close < ema20_1h)
        at_pullback_long  = near_ema or wick_bounce_l
        at_pullback_short = near_ema or wick_reject_s

        # ── MTF bias pre-filter ───────────────────────────────────────────
        comp_pct, bias_label = self.compute_mtf_bias(
            candles, mtf_candles,
            ema_fast=20, ema_slow=50, rsi_period=14, rsi_bull=55.0, rsi_bear=45.0,
        )
        long_bias_ok  = comp_pct >  self.bias_gate
        short_bias_ok = comp_pct < -self.bias_gate
        if not (long_bias_ok or short_bias_ok):
            return Signal(
                SignalType.HOLD, self.symbol, current_price, 0,
                reason=(f"[TrendCont] bias weak: comp={comp_pct:.0f} ({bias_label}) "
                        f"need >|{self.bias_gate:.0f}|"),
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
        atr_v   = float(atr_a[-2])

        if any(np.isnan(v) for v in [ema20_b, ema9_b, rsi_b, atr_v]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[TrendCont] 15m indicators not ready")

        vol_ok = volma_b > 0 and vol_b >= volma_b * self.vol_mult

        def _meta(side: str) -> dict:
            return self.risk_metadata(
                current_price, atr_v, side,
                sl_atr=self.sl_atr, tp1_atr=self.tp1_atr, tp2_atr=self.tp2_atr,
                sl_min_pct=self.sl_min_pct, sl_max_pct=self.sl_max_pct,
                risk_pct=self.risk_pct, partial_pct=self.partial_pct,
            )

        # ── BUY: 4H up + ST green + 1H up + pullback + bias + cond ─────────
        if macro_up and st_green and mid_up and at_pullback_long and long_bias_ok:
            c1 = close_b > ema9_b
            c2 = close_b > ema20_b
            c3 = self.rsi_min_buy <= rsi_b <= self.rsi_max_buy
            c4 = vol_ok
            met = sum([c1, c2, c3, c4])
            if met >= self.min_entry_cond:
                meta = _meta("long")
                return Signal(
                    SignalType.BUY, self.symbol, current_price, amount=0.08,
                    reason=(f"[TrendCont] 4H↑ ST🟢 1H↑ bias={comp_pct:.0f} EMA20pull "
                            f"cond={met}/4 RSI={rsi_b:.0f} TP1={meta['tp1']} TP2={meta['tp2']}"),
                    confidence=0.65 + met * 0.02,
                    metadata=meta,
                )

        # ── SELL: 4H down + ST red + 1H down + rally + bias + cond ─────────
        if macro_down and st_red and mid_down and at_pullback_short and short_bias_ok:
            c1 = close_b < ema9_b
            c2 = close_b < ema20_b
            c3 = self.rsi_min_sell <= rsi_b <= self.rsi_max_sell
            c4 = vol_ok
            met = sum([c1, c2, c3, c4])
            if met >= self.min_entry_cond:
                meta = _meta("short")
                return Signal(
                    SignalType.SELL, self.symbol, current_price, amount=0.08,
                    reason=(f"[TrendCont] 4H↓ ST🔴 1H↓ bias={comp_pct:.0f} EMA20resist "
                            f"cond={met}/4 RSI={rsi_b:.0f} TP1={meta['tp1']} TP2={meta['tp2']}"),
                    confidence=0.65 + met * 0.02,
                    metadata=meta,
                )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            reason=(f"[TrendCont] "
                    f"4H={'↑' if macro_up else '↓' if macro_down else '→'} "
                    f"ST={'🟢' if st_green else '🔴' if st_red else '—'} "
                    f"1H={'↑' if mid_up else '↓' if mid_down else '→'} "
                    f"bias={comp_pct:.0f} pull-L={at_pullback_long} pull-S={at_pullback_short}"),
        )
