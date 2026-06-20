"""
Mean Reversion Strategy — Futures Long & Short  (v2 — High-WR / TP1+TP2)
=======================================================================
Counter-trend bounce at extremes, but only in non-violent markets.

  BUY  (long)  : 4H "up"   + ADX(4H) ≤ adx_cap + 1H min_conditions/4 (RSI oversold bounce)
  SELL (short) : 4H "down" + ADX(4H) ≤ adx_cap + 1H min_conditions/4 (RSI overbought drop)

4H trend: EMA20 vs EMA50 + slope.

ADX filter (NEW): a counter-trend entry is REFUSED (HOLD) when ADX(14) on 4H
> adx_cap (default 30) — a strong trend is too dangerous to fade.

1H conditions (min_conditions/4, default 3 — raised from 2 for selectivity):
  1. RSI ≤ oversold on BOTH current AND prev bar, bouncing up
  2. MACD histogram flip up / continuing positive
  3. Bollinger lower-band bounce
  4. Volume spike ≥ vol_spike_mult × MA20

Risk (Global): Initial SL 1.5×ATR, TP1 1.2×ATR (close 50% → SL→BE), TP2 2.2×ATR.
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class MeanReversionStrategy(BaseStrategy):

    MTF_TIMEFRAMES = ["4h"]

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.rsi_period     = self.params.get("rsi_period",     14)
        self.rsi_oversold   = self.params.get("rsi_oversold",  45.0)
        self.rsi_overbought = self.params.get("rsi_overbought",55.0)
        self.macd_fast      = self.params.get("macd_fast",      12)
        self.macd_slow      = self.params.get("macd_slow",      26)
        self.macd_sig       = self.params.get("macd_signal",     9)
        self.bb_period      = self.params.get("bb_period",      20)
        self.bb_std         = self.params.get("bb_std",         2.0)
        self.vol_period     = self.params.get("vol_period",     20)
        self.vol_spike_mult = self.params.get("vol_spike_mult", 1.2)
        self.min_conditions = self.params.get("min_conditions",   3)  # ↑ from 2 — stricter
        self.adx_period     = self.params.get("adx_period",     14)
        self.adx_cap        = self.params.get("adx_cap",       50.0)  # NEW: no fade if 4H ADX > 50
        self.atr_period     = self.params.get("atr_period",     14)
        # Global-Risk multiples (grid-tuned: SL 1.2 / TP1 1.5 / TP2 3.0 ATR)
        self.sl_atr         = self.params.get("sl_atr",         1.2)
        self.tp1_atr        = self.params.get("tp1_atr",        1.5)
        self.tp2_atr        = self.params.get("tp2_atr",        3.0)
        self.sl_min_pct     = self.params.get("sl_min_pct",   0.010)
        self.sl_max_pct     = self.params.get("sl_max_pct",   0.040)
        self.risk_pct       = self.params.get("risk_pct",      0.02)
        self.partial_pct    = self.params.get("partial_pct",    0.5)
        self.trend_ema_fast = self.params.get("trend_ema_fast", 20)
        self.trend_ema_slow = self.params.get("trend_ema_slow", 50)

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        mtf        = mtf_candles or {}
        candles_4h = mtf.get("4h", [])

        if len(candles) < 60 or len(candles_4h) < 55:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[MeanRev] Not enough data")

        closes    = [c.close for c in candles]
        vols      = [c.volume for c in candles]
        closes_4h = [c.close for c in candles_4h]

        # ── 4H trend: EMA20 vs EMA50 + slope ──────────────────────────────
        ef4 = self.ema(closes_4h, self.trend_ema_fast)
        es4 = self.ema(closes_4h, self.trend_ema_slow)
        if np.isnan(ef4[-1]) or np.isnan(es4[-1]) or np.isnan(ef4[-2]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[MeanRev] 4H EMA not ready")
        slope_up = ef4[-1] > ef4[-2]
        if ef4[-1] > es4[-1] and slope_up:
            trend_4h = "up"
        elif ef4[-1] < es4[-1] and not slope_up:
            trend_4h = "down"
        else:
            trend_4h = "flat"

        # ── ADX(4H) filter: refuse to fade a violent trend ────────────────
        adx4, _, _ = self.adx(candles_4h, self.adx_period)
        adx_v = float(adx4[-2]) if not np.isnan(adx4[-2]) else 0.0
        if adx_v > self.adx_cap:
            return Signal(
                SignalType.HOLD, self.symbol, current_price, 0,
                reason=f"[MeanRev] 4H ADX={adx_v:.0f} > {self.adx_cap:.0f} — trend too strong to fade",
            )

        # ── 1H indicators (last closed = [-2], prev = [-3]) ───────────────
        rsi_a               = self.rsi(closes, self.rsi_period)
        _, _, macd_hist     = self.macd(closes, self.macd_fast, self.macd_slow, self.macd_sig)
        bb_upper, _, bb_low = self.bollinger_bands(closes, self.bb_period, self.bb_std)
        vol_ma              = self.sma(vols, self.vol_period)
        atr_a               = self.atr(candles, self.atr_period)

        rsi_c   = float(rsi_a[-2]);     rsi_p   = float(rsi_a[-3])
        hist_c  = float(macd_hist[-2]); hist_p  = float(macd_hist[-3])
        bbl_c   = float(bb_low[-2]);    bbl_p   = float(bb_low[-3])
        bbu_c   = float(bb_upper[-2]);  bbu_p   = float(bb_upper[-3])
        close_c = closes[-2];           close_p = closes[-3]
        vol_c   = vols[-2];             volma_c = float(vol_ma[-2])
        atr_v   = float(atr_a[-2])  # last closed bar's ATR

        if any(np.isnan(v) for v in [rsi_c, rsi_p, hist_c, bbl_c, bbu_c, volma_c, atr_v]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[MeanRev] Indicators not ready")

        vol_ok = volma_c > 0 and vol_c >= volma_c * self.vol_spike_mult

        def _meta(side: str) -> dict:
            return self.risk_metadata(
                current_price, atr_v, side,
                sl_atr=self.sl_atr, tp1_atr=self.tp1_atr, tp2_atr=self.tp2_atr,
                sl_min_pct=self.sl_min_pct, sl_max_pct=self.sl_max_pct,
                risk_pct=self.risk_pct, partial_pct=self.partial_pct,
            )

        # ── BUY (long): 4H "up" + ADX ok + min_conditions/4 ───────────────
        if trend_4h == "up":
            c1  = rsi_c <= self.rsi_oversold and rsi_p <= self.rsi_oversold and rsi_c > rsi_p
            c2  = (hist_p < 0 and hist_c > 0) or (hist_c > 0 and hist_c > hist_p and not np.isnan(hist_p))
            c3  = (close_p <= bbl_p) and (close_c > bbl_c)
            met = sum([c1, c2, c3, vol_ok])
            if met >= self.min_conditions:
                meta = _meta("long")
                return Signal(
                    SignalType.BUY, self.symbol, current_price, amount=0.08,
                    reason=(f"[MeanRev] 4H↑ ADX={adx_v:.0f} RSI={rsi_c:.0f} "
                            f"cond={met}/4 TP1={meta['tp1']} TP2={meta['tp2']}"),
                    confidence=min(0.55 + met * 0.08, 0.87),
                    metadata=meta,
                )

        # ── SELL (short): 4H "down" + ADX ok + min_conditions/4 ───────────
        if trend_4h == "down":
            c1  = rsi_c >= self.rsi_overbought and rsi_p >= self.rsi_overbought and rsi_c < rsi_p
            c2  = (hist_p > 0 and hist_c < 0) or (hist_c < 0 and hist_c < hist_p and not np.isnan(hist_p))
            c3  = (close_p >= bbu_p) and (close_c < bbu_c)
            met = sum([c1, c2, c3, vol_ok])
            if met >= self.min_conditions:
                meta = _meta("short")
                return Signal(
                    SignalType.SELL, self.symbol, current_price, amount=0.08,
                    reason=(f"[MeanRev] 4H↓ ADX={adx_v:.0f} RSI={rsi_c:.0f} "
                            f"cond={met}/4 TP1={meta['tp1']} TP2={meta['tp2']}"),
                    confidence=min(0.55 + met * 0.08, 0.87),
                    metadata=meta,
                )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            reason=f"[MeanRev] 4H={trend_4h} ADX={adx_v:.0f} RSI={rsi_c:.0f} cond not met",
        )
