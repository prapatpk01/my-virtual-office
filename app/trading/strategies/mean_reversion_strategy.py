"""
Mean Reversion Strategy — Futures Long & Short
===============================================
Port of mean_reversion style from unified_trading_bot.py.

  BUY  (long)  : 4H NOT "down" + 1H 3/4 conditions (oversold 2-bar confirmed)
  SELL (short) : 4H NOT "up"   + 1H 3/4 conditions (overbought 2-bar confirmed)

4H trend: EMA20 > EMA50 + positive slope → "up", EMA20 < EMA50 + falling slope → "down"

1H conditions (min_conditions/4, default 3):
  1. RSI ≤ oversold on BOTH current AND prev bar, bouncing up   [2-bar RSI]
  2. MACD histogram flip up or continuing positive
  3. Bollinger Band lower bounce (prev close ≤ lower, cur close > lower)
  4. Volume spike ≥ vol_spike_mult × MA20

SL/TP: ATR-based, clamped to [sl_min_pct, sl_max_pct]
  sl_mult=1.2, tp_mult=2.0 → R:R ≈ 1:1.67  (break-even WR 37.5%)
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class MeanReversionStrategy(BaseStrategy):

    MTF_TIMEFRAMES = ["4h"]

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.rsi_period     = self.params.get("rsi_period",     14)
        self.rsi_oversold   = self.params.get("rsi_oversold",  35.0)
        self.rsi_overbought = self.params.get("rsi_overbought",65.0)
        self.macd_fast      = self.params.get("macd_fast",      12)
        self.macd_slow      = self.params.get("macd_slow",      26)
        self.macd_sig       = self.params.get("macd_signal",     9)
        self.bb_period      = self.params.get("bb_period",      20)
        self.bb_std         = self.params.get("bb_std",         2.0)
        self.vol_period     = self.params.get("vol_period",     20)
        self.vol_spike_mult = self.params.get("vol_spike_mult", 1.3)
        self.min_conditions = self.params.get("min_conditions",   3)
        self.atr_period     = self.params.get("atr_period",     14)
        self.sl_mult        = self.params.get("sl_mult",        1.2)
        self.tp_mult        = self.params.get("tp_mult",        2.0)
        self.sl_min_pct     = self.params.get("sl_min_pct",   0.004)
        self.sl_max_pct     = self.params.get("sl_max_pct",   0.025)
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
        atr_v   = float(atr_a[-1])

        if any(np.isnan(v) for v in [rsi_c, rsi_p, hist_c, bbl_c, bbu_c, volma_c, atr_v]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[MeanRev] Indicators not ready")

        vol_ok = volma_c > 0 and vol_c >= volma_c * self.vol_spike_mult
        rr     = self.tp_mult / max(self.sl_mult, 1e-9)

        def _sl_tp(side: str) -> tuple[float, float]:
            raw  = atr_v * self.sl_mult
            dist = max(current_price * self.sl_min_pct,
                       min(raw, current_price * self.sl_max_pct))
            tp_d = atr_v * self.tp_mult
            if side == "long":
                return round(current_price - dist, 2), round(current_price + tp_d, 2)
            return round(current_price + dist, 2), round(current_price - tp_d, 2)

        # ── BUY (long): 4H NOT "down" + 3/4 1H conditions ────────────────
        if trend_4h != "down":
            # 2-bar confirmation: both bars must be oversold, cur bouncing
            c1  = (rsi_p <= self.rsi_oversold) and (rsi_c <= self.rsi_oversold) and (rsi_c > rsi_p)
            c2  = (hist_p < 0 and hist_c > 0) or (hist_c > 0 and hist_c > hist_p and not np.isnan(hist_p))
            c3  = (close_p <= bbl_p) and (close_c > bbl_c)
            met = sum([c1, c2, c3, vol_ok])
            if met >= self.min_conditions:
                sl, tp = _sl_tp("long")
                return Signal(
                    SignalType.BUY, self.symbol, current_price,
                    amount=0.08,
                    reason=(f"[MeanRev] 4H={trend_4h} RSI={rsi_c:.0f}(2bar) "
                            f"MACD={'✓' if c2 else '✗'} BB={'✓' if c3 else '✗'} "
                            f"cond={met}/4 RR=1:{rr:.2f}"),
                    confidence=min(0.55 + met * 0.08, 0.87),
                    metadata={"stop_loss": sl, "take_profit": tp, "atr": atr_v},
                )

        # ── SELL (short): 4H NOT "up" + 3/4 1H conditions ────────────────
        if trend_4h != "up":
            c1  = (rsi_p >= self.rsi_overbought) and (rsi_c >= self.rsi_overbought) and (rsi_c < rsi_p)
            c2  = (hist_p > 0 and hist_c < 0) or (hist_c < 0 and hist_c < hist_p and not np.isnan(hist_p))
            c3  = (close_p >= bbu_p) and (close_c < bbu_c)
            met = sum([c1, c2, c3, vol_ok])
            if met >= self.min_conditions:
                sl, tp = _sl_tp("short")
                return Signal(
                    SignalType.SELL, self.symbol, current_price,
                    amount=0.08,
                    reason=(f"[MeanRev] 4H={trend_4h} RSI={rsi_c:.0f}(2bar) "
                            f"MACD={'✓' if c2 else '✗'} BB={'✓' if c3 else '✗'} "
                            f"cond={met}/4 RR=1:{rr:.2f}"),
                    confidence=min(0.55 + met * 0.08, 0.87),
                    metadata={"stop_loss": sl, "take_profit": tp, "atr": atr_v},
                )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            reason=f"[MeanRev] 4H={trend_4h} RSI={rsi_c:.0f} cond not met",
        )
