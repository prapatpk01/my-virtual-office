"""
Smart Money Strategy — MTF Momentum  (v2 — stronger filters / TP1+TP2)
=====================================================================
MTF momentum confirmed by real volume flow (OBV).

  BUY  (long)  : comp_pct > +bias_threshold AND EMA9>EMA21 (15m) AND RSI neutral
                 AND volume ≥ vol_mult×MA20 AND OBV > EMA20(OBV)   [money flowing in]
  SELL (short) : comp_pct < -bias_threshold AND EMA9<EMA21 (15m) AND RSI neutral
                 AND volume ≥ vol_mult×MA20 AND OBV < EMA20(OBV)   [money flowing out]

Changes vs v1:
  • bias_threshold 15 → 40  (skip directionless / sideways chop)
  • NEW OBV vs EMA20(OBV) confirmation — only trade when big money agrees

Risk (Global): Initial SL 1.5×ATR, TP1 1.2×ATR (close 50% → SL→BE), TP2 2.2×ATR.
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class SmartMoneyStrategy(BaseStrategy):

    MTF_TIMEFRAMES = ["1h", "4h"]

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.ema_fast       = self.params.get("ema_fast",        9)
        self.ema_slow       = self.params.get("ema_slow",       21)
        self.rsi_period     = self.params.get("rsi_period",     14)
        self.vol_period     = self.params.get("vol_period",     20)
        self.vol_mult       = self.params.get("vol_mult",       1.0)
        self.bias_threshold = self.params.get("bias_threshold", 40.0)  # ↑ from 15 — skip chop
        self.obv_ema        = self.params.get("obv_ema",          20)  # NEW: EMA of OBV
        self.atr_period     = self.params.get("atr_period",     14)
        # Global-Risk multiples (grid-tuned: SL 1.2 / TP1 1.5 / TP2 3.0 ATR)
        self.sl_atr         = self.params.get("sl_atr",         1.2)
        self.tp1_atr        = self.params.get("tp1_atr",        1.5)
        self.tp2_atr        = self.params.get("tp2_atr",        3.0)
        self.sl_min_pct     = self.params.get("sl_min_pct",   0.010)
        self.sl_max_pct     = self.params.get("sl_max_pct",   0.050)
        self.risk_pct       = self.params.get("risk_pct",      0.02)
        self.partial_pct    = self.params.get("partial_pct",    0.5)

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        mtf = mtf_candles or {}

        if len(candles) < 55 or len(mtf.get("1h", [])) < 55 or len(mtf.get("4h", [])) < 55:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[SmartMoney] Not enough MTF data")

        closes_15m = [c.close for c in candles]
        vols_15m   = [c.volume for c in candles]

        # ── MTF composite bias (15m+1H+4H) ───────────────────────────────
        comp_pct, bias_label = self.compute_mtf_bias(
            candles, mtf,
            ema_fast=20, ema_slow=50, rsi_period=14, rsi_bull=55.0, rsi_bear=45.0,
        )

        long_ok  = comp_pct >  self.bias_threshold
        short_ok = comp_pct < -self.bias_threshold

        if not (long_ok or short_ok):
            return Signal(
                SignalType.HOLD, self.symbol, current_price, 0,
                reason=(f"[SmartMoney] Bias too weak: comp={comp_pct:.0f} ({bias_label}) "
                        f"need >|{self.bias_threshold:.0f}|"),
            )

        # ── 15m indicators (last closed bar = [-2]) ───────────────────────
        ef9   = self.ema(closes_15m, self.ema_fast)
        ef21  = self.ema(closes_15m, self.ema_slow)
        rsi_a = self.rsi(closes_15m, self.rsi_period)
        volma = self.sma(vols_15m,   self.vol_period)
        atr_a = self.atr(candles,    self.atr_period)

        # ── OBV flow confirmation (NEW) ───────────────────────────────────
        obv_a    = self.obv(candles)
        obv_emaA = self.ema(list(obv_a), self.obv_ema)
        obv_b    = float(obv_a[-2])
        obv_e_b  = float(obv_emaA[-2])
        obv_up   = (not np.isnan(obv_e_b)) and obv_b > obv_e_b
        obv_down = (not np.isnan(obv_e_b)) and obv_b < obv_e_b

        ema9_b  = float(ef9[-2])
        ema21_b = float(ef21[-2])
        rsi_b   = float(rsi_a[-2])
        vol_b   = vols_15m[-2]
        volma_b = float(volma[-2])
        atr_v   = float(atr_a[-2])

        if any(np.isnan(v) for v in [ema9_b, ema21_b, rsi_b, atr_v]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[SmartMoney] 15m indicators not ready")

        rsi_neutral = 35.0 <= rsi_b <= 65.0
        vol_ok      = volma_b > 0 and vol_b >= volma_b * self.vol_mult

        def _meta(side: str) -> dict:
            return self.risk_metadata(
                current_price, atr_v, side,
                sl_atr=self.sl_atr, tp1_atr=self.tp1_atr, tp2_atr=self.tp2_atr,
                sl_min_pct=self.sl_min_pct, sl_max_pct=self.sl_max_pct,
                risk_pct=self.risk_pct, partial_pct=self.partial_pct,
            )

        # ── BUY: bias + EMA9>EMA21 + RSI neutral + volume + OBV up ─────────
        if long_ok and ema9_b > ema21_b and rsi_neutral and vol_ok and obv_up:
            meta = _meta("long")
            return Signal(
                SignalType.BUY, self.symbol, current_price, amount=0.08,
                reason=(f"[SmartMoney] LONG comp={comp_pct:.0f} EMA9>EMA21 OBV↑ "
                        f"RSI={rsi_b:.0f} TP1={meta['tp1']} TP2={meta['tp2']}"),
                confidence=min(0.50 + abs(comp_pct) / 200.0, 0.85),
                metadata=meta,
            )

        # ── SELL: bias + EMA9<EMA21 + RSI neutral + volume + OBV down ──────
        if short_ok and ema9_b < ema21_b and rsi_neutral and vol_ok and obv_down:
            meta = _meta("short")
            return Signal(
                SignalType.SELL, self.symbol, current_price, amount=0.08,
                reason=(f"[SmartMoney] SHORT comp={comp_pct:.0f} EMA9<EMA21 OBV↓ "
                        f"RSI={rsi_b:.0f} TP1={meta['tp1']} TP2={meta['tp2']}"),
                confidence=min(0.50 + abs(comp_pct) / 200.0, 0.85),
                metadata=meta,
            )

        ema_dir = "EMA9>EMA21" if ema9_b > ema21_b else "EMA9<EMA21"
        obv_dir = "OBV↑" if obv_up else "OBV↓" if obv_down else "OBV→"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            reason=(f"[SmartMoney] comp={comp_pct:.0f} {ema_dir} {obv_dir} "
                    f"RSI={rsi_b:.0f} vol={'ok' if vol_ok else 'low'} — not fully met"),
        )
