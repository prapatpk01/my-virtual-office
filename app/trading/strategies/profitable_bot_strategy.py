"""
Profitable Bot Strategy — Futures Long & Short
===============================================
Adapted from profitable.py, extended to support both long and short positions.

3-layer filter:
  Layer 1 — 4H trend guard (EMA20 vs EMA50 position)
  Layer 2 — 1H entry: min_cond / 4 conditions required (default 2/4)
  Layer 3 — ATR-based SL/TP

sl_mult=1.5, tp_mult=1.875 → R:R = 1:1.25 (break-even WR = 44.4%)

BUY  (long)  : 4H NOT "down" + 1H oversold bounce (min_cond/4 conditions met)
SELL (short) : 4H "down"     + 1H overbought rejection (min_cond/4 conditions met)

1H conditions (any min_cond of 4):
  1. RSI oversold / overbought (single bar, adjustable threshold)
  2. MACD histogram direction crossover or continuation
  3. Bollinger Band bounce off lower / rejection off upper band
  4. Volume spike ≥ vol_mult × MA20

MTF candles required: 4h (fetched automatically via MTF_TIMEFRAMES).
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class ProfitableBotStrategy(BaseStrategy):

    MTF_TIMEFRAMES = ["4h"]

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.rsi_period      = self.params.get("rsi_period",       14)
        self.rsi_oversold    = self.params.get("rsi_oversold",    42.0)  # ↑ from 35 — more signals
        self.rsi_overbought  = self.params.get("rsi_overbought",  58.0)  # ↓ from 65 — more signals
        self.macd_fast       = self.params.get("macd_fast",        12)
        self.macd_slow       = self.params.get("macd_slow",        26)
        self.macd_sig        = self.params.get("macd_signal",       9)
        self.bb_period       = self.params.get("bb_period",        20)
        self.bb_std          = self.params.get("bb_std",           2.0)
        self.vol_period      = self.params.get("vol_period",       20)
        self.vol_mult        = self.params.get("vol_mult",         1.2)
        self.min_cond        = self.params.get("min_conditions",     2)  # ↓ from 3 — 2/4 needed
        self.atr_period      = self.params.get("atr_period",       14)
        self.sl_mult         = self.params.get("sl_mult",          1.5)
        self.tp_mult         = self.params.get("tp_mult",        1.875)  # SL×1.25 → R:R 1:1.25
        self.sl_min_pct      = self.params.get("sl_min_pct",      0.010)
        self.sl_max_pct      = self.params.get("sl_max_pct",      0.070)
        self.trend_ema_fast  = self.params.get("trend_ema_fast",   20)
        self.trend_ema_slow  = self.params.get("trend_ema_slow",   50)

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        mtf        = mtf_candles or {}
        candles_4h = mtf.get("4h", [])

        if len(candles) < 60 or len(candles_4h) < 55:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[ProfBot] Not enough data")

        closes    = [c.close for c in candles]
        vols      = [c.volume for c in candles]
        closes_4h = [c.close for c in candles_4h]

        # ── 4H trend guard (position only, no slope required) ────────────────
        ef4 = self.ema(closes_4h, self.trend_ema_fast)
        es4 = self.ema(closes_4h, self.trend_ema_slow)
        if np.isnan(ef4[-1]) or np.isnan(es4[-1]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[ProfBot] 4H EMA not ready")
        # Simple position-based trend (removed slope requirement for more signals)
        if ef4[-1] > es4[-1]:
            trend_4h = "up"
        elif ef4[-1] < es4[-1]:
            trend_4h = "down"
        else:
            trend_4h = "flat"

        # ── 1H indicators (last closed = [-2], prev = [-3]) ───────────────────
        rsi_a               = self.rsi(closes, self.rsi_period)
        _, _, macd_hist     = self.macd(closes, self.macd_fast, self.macd_slow, self.macd_sig)
        bb_upper, _, bb_low = self.bollinger_bands(closes, self.bb_period, self.bb_std)
        vol_ma              = self.sma(vols, self.vol_period)
        atr_a               = self.atr(candles, self.atr_period)

        rsi_c   = float(rsi_a[-2]);    rsi_p   = float(rsi_a[-3])
        hist_c  = float(macd_hist[-2]); hist_p  = float(macd_hist[-3])
        bbl_c   = float(bb_low[-2]);   bbl_p   = float(bb_low[-3])
        bbu_c   = float(bb_upper[-2]); bbu_p   = float(bb_upper[-3])
        close_c = closes[-2];          close_p = closes[-3]
        vol_c   = vols[-2]
        volma_c = float(vol_ma[-2])
        atr_v   = float(atr_a[-1])

        if any(np.isnan(v) for v in [rsi_c, rsi_p, hist_c, bbl_c, bbu_c, volma_c, atr_v]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[ProfBot] Indicators not ready")

        vol_ok = volma_c > 0 and vol_c >= volma_c * self.vol_mult
        rr     = self.tp_mult / max(self.sl_mult, 1e-9)

        def _sl_tp(side: str) -> tuple[float, float]:
            raw  = atr_v * self.sl_mult
            dist = max(current_price * self.sl_min_pct,
                       min(raw, current_price * self.sl_max_pct))
            if side == "long":
                return round(current_price - dist, 2), round(current_price + atr_v * self.tp_mult, 2)
            return round(current_price + dist, 2), round(current_price - atr_v * self.tp_mult, 2)

        # ── BUY (long): 4H up/flat + 1H oversold bounce ──────────────────────
        if trend_4h != "down":
            # Single-bar RSI check (not 2-bar, more lenient)
            rsi_bull   = rsi_c <= self.rsi_oversold and rsi_c > rsi_p   # oversold + bouncing
            macd_bull  = ((hist_p < 0 and hist_c > 0) or
                          (hist_c > 0 and hist_c > hist_p and not np.isnan(hist_p)))
            bb_bounce  = (close_p <= bbl_p) and (close_c > bbl_c)
            met = sum([rsi_bull, macd_bull, bb_bounce, vol_ok])
            if met >= self.min_cond:
                sl, tp = _sl_tp("long")
                return Signal(
                    SignalType.BUY, self.symbol, current_price,
                    amount=0.08,
                    reason=(f"[ProfBot] 4H={trend_4h} RSI={rsi_c:.0f} "
                            f"MACD={'✓' if macd_bull else '✗'} BB={'✓' if bb_bounce else '✗'} "
                            f"cond={met}/4 RR=1:{rr:.2f}"),
                    confidence=min(0.55 + met * 0.07, 0.85),
                    metadata={"stop_loss": sl, "take_profit": tp, "atr": atr_v},
                )

        # ── SELL (short): 4H down + 1H overbought rejection ──────────────────
        if trend_4h == "down":
            rsi_bear   = rsi_c >= self.rsi_overbought and rsi_c < rsi_p   # overbought + dropping
            macd_bear  = ((hist_p > 0 and hist_c < 0) or
                          (hist_c < 0 and hist_c < hist_p and not np.isnan(hist_p)))
            bb_reject  = (close_p >= bbu_p) and (close_c < bbu_c)
            met = sum([rsi_bear, macd_bear, bb_reject, vol_ok])
            if met >= self.min_cond:
                sl, tp = _sl_tp("short")
                return Signal(
                    SignalType.SELL, self.symbol, current_price,
                    amount=0.08,
                    reason=(f"[ProfBot] 4H=down RSI={rsi_c:.0f} "
                            f"MACD={'✓' if macd_bear else '✗'} BB={'✓' if bb_reject else '✗'} "
                            f"cond={met}/4 RR=1:{rr:.2f}"),
                    confidence=min(0.55 + met * 0.07, 0.85),
                    metadata={"stop_loss": sl, "take_profit": tp, "atr": atr_v},
                )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            reason=f"[ProfBot] 4H={trend_4h} RSI={rsi_c:.0f} cond not met",
        )
