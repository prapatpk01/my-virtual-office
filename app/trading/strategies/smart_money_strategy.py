"""
Smart Money Strategy — MTF Momentum (rewritten)
================================================
Replaces the BOS/CHoCH approach with a clean MTF Momentum strategy.

Entry logic (3-layer MTF alignment):
  BUY  (long)  : comp_pct > +bias_threshold AND RSI in [35,65]
                 AND EMA9 > EMA21 (15m) AND volume >= vol_mult x MA20
  SELL (short) : comp_pct < -bias_threshold AND RSI in [35,65]
                 AND EMA9 < EMA21 (15m) AND volume >= vol_mult x MA20

comp_pct from compute_mtf_bias() ranges -100 to +100 and captures
the composite 15m+1H+4H trend alignment.

SL/TP: ATR-based, clamped to [sl_min_pct, sl_max_pct]
  sl_mult=1.5, tp_mult=2.5 → R:R 1:1.67  (break-even WR 37.5%)
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
        self.bias_threshold = self.params.get("bias_threshold", 15.0)  # ↓ from 20 — grid-optimised
        self.atr_period     = self.params.get("atr_period",     14)
        self.sl_mult        = self.params.get("sl_mult",        1.2)   # Case3 optimised ↓ from 1.5
        self.tp_mult        = self.params.get("tp_mult",        1.5)   # Case3 optimised ↓ from 2.5
        self.sl_min_pct     = self.params.get("sl_min_pct",   0.010)  # 1.0% min SL
        self.sl_max_pct     = self.params.get("sl_max_pct",   0.050)

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
                        f"need >{self.bias_threshold:.0f} long / <-{self.bias_threshold:.0f} short"),
            )

        # ── 15m indicators (last closed bar = [-2]) ───────────────────────
        ef9   = self.ema(closes_15m, self.ema_fast)
        ef21  = self.ema(closes_15m, self.ema_slow)
        rsi_a = self.rsi(closes_15m, self.rsi_period)
        volma = self.sma(vols_15m,   self.vol_period)
        atr_a = self.atr(candles,    self.atr_period)

        ema9_b  = float(ef9[-2])
        ema21_b = float(ef21[-2])
        rsi_b   = float(rsi_a[-2])
        vol_b   = vols_15m[-2]
        volma_b = float(volma[-2])
        atr_v   = float(atr_a[-1])

        if any(np.isnan(v) for v in [ema9_b, ema21_b, rsi_b, atr_v]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[SmartMoney] 15m indicators not ready")

        rsi_neutral = 35.0 <= rsi_b <= 65.0
        vol_ok      = volma_b > 0 and vol_b >= volma_b * self.vol_mult
        rr          = self.tp_mult / max(self.sl_mult, 1e-9)

        def _sl_tp(side: str) -> tuple[float, float]:
            raw  = atr_v * self.sl_mult
            dist = max(current_price * self.sl_min_pct,
                       min(raw, current_price * self.sl_max_pct))
            tp_d = dist * (self.tp_mult / self.sl_mult)  # scale TP with actual dist to maintain R:R
            if side == "long":
                return round(current_price - dist, 2), round(current_price + tp_d, 2)
            return round(current_price + dist, 2), round(current_price - tp_d, 2)

        # ── BUY: strong bullish bias + EMA9>EMA21 + RSI neutral + volume ─
        if long_ok and ema9_b > ema21_b and rsi_neutral and vol_ok:
            sl, tp = _sl_tp("long")
            return Signal(
                SignalType.BUY, self.symbol, current_price,
                amount=0.08,
                reason=(f"[SmartMoney] LONG comp={comp_pct:.0f} ({bias_label}) "
                        f"EMA9>EMA21 RSI={rsi_b:.0f} RR=1:{rr:.2f}"),
                confidence=min(0.50 + abs(comp_pct) / 200.0, 0.85),
                metadata={"stop_loss": sl, "take_profit": tp, "atr": atr_v},
            )

        # ── SELL: strong bearish bias + EMA9<EMA21 + RSI neutral + volume ─
        if short_ok and ema9_b < ema21_b and rsi_neutral and vol_ok:
            sl, tp = _sl_tp("short")
            return Signal(
                SignalType.SELL, self.symbol, current_price,
                amount=0.08,
                reason=(f"[SmartMoney] SHORT comp={comp_pct:.0f} ({bias_label}) "
                        f"EMA9<EMA21 RSI={rsi_b:.0f} RR=1:{rr:.2f}"),
                confidence=min(0.50 + abs(comp_pct) / 200.0, 0.85),
                metadata={"stop_loss": sl, "take_profit": tp, "atr": atr_v},
            )

        ema_dir = "EMA9>EMA21" if ema9_b > ema21_b else "EMA9<EMA21"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            reason=(f"[SmartMoney] comp={comp_pct:.0f} ({bias_label}) "
                    f"{ema_dir} RSI={rsi_b:.0f} vol={'ok' if vol_ok else 'low'} "
                    f"— conditions not fully met"),
        )
