"""
SpotMaster1H — 7-layer multi-timeframe entry for spot long trading.

Layer 1 (1D)  Regime:    EMA50>EMA200  OR  RSI14>50
Layer 2 (4H)  Trend:     EMA20>EMA50  AND  EMA50>EMA200  AND  Close>EMA20
Layer 3 (4H)  Strength:  ADX14 > 18  AND  ADX14 rising
Layer 4 (4H)  Pullback:  1H Low <= EMA20(4H)  OR  1H Low <= BBMid(4H,20)
Layer 5 (1H)  Score:     ≥6/7  (EMA5>EMA9 | Close>HMA16 | RSI>50 | ROC9>0 |
                                 Vol>VolMA20 | MACDhist>0 | Close>PrevHigh)
Layer 6 (1H)  Momentum:  ≥2/3  (KST>Signal | MACDhist rising | ROC3 rising)
Layer 7       Health:    Score > 70 / 100
                           Trend:    EMA20(4H)>EMA50 +20 | EMA50(4H)>EMA200 +15
                                     EMA50(1D)>EMA200 +20
                           Momentum: ROC>0 +10 | MACD>0 +10 | RSI 55-75 +10
                           Volume:   Vol expansion +15

Exit:
  TP1 25% @ 1R  |  TP2 25% @ 2R (primary OCO)  |  TP3 50% trailing EMA20(4H)
  Hard exit: EMA20(4H) < EMA50(4H)  OR  Close(4H) < EMA50(4H) ×2 bars
"""
import math

from .base import BaseStrategy, Signal, SignalType

_WARMUP = 60   # minimum 1H bars (KST needs ~40, HMA16 warmup)


# ── Module-level indicator helpers ────────────────────────────────────────────

def _roc(closes: list, period: int) -> list:
    """Rate of Change in percent: (close - close[n-period]) / close[n-period] * 100."""
    n = len(closes)
    out = [float('nan')] * n
    for i in range(period, n):
        prev = closes[i - period]
        if abs(prev) > 1e-10:
            out[i] = (closes[i] - prev) / prev * 100.0
    return out


def _sma_list(arr: list, period: int) -> list:
    n = len(arr)
    out = [float('nan')] * n
    for i in range(period - 1, n):
        vals = [v for v in arr[i - period + 1:i + 1] if not math.isnan(v)]
        if len(vals) == period:
            out[i] = sum(vals) / period
    return out


def _kst(closes: list) -> tuple:
    """KST oscillator (Martin Pring). Returns (kst_arr, signal_arr)."""
    rcma1 = _sma_list(_roc(closes, 10), 10)
    rcma2 = _sma_list(_roc(closes, 13), 13)
    rcma3 = _sma_list(_roc(closes, 15), 15)
    rcma4 = _sma_list(_roc(closes, 20), 20)
    n = len(closes)
    kst_arr = [float('nan')] * n
    for i in range(n):
        vs = [rcma1[i], rcma2[i], rcma3[i], rcma4[i]]
        if not any(math.isnan(v) for v in vs):
            kst_arr[i] = vs[0] + 2.0 * vs[1] + 3.0 * vs[2] + 4.0 * vs[3]
    return kst_arr, _sma_list(kst_arr, 9)


# ── Strategy ──────────────────────────────────────────────────────────────────

class SpotMaster1H(BaseStrategy):
    """7-layer spot-long strategy. Requires mtf_candles['4h'] and mtf_candles['1d']."""

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.sl_atr        = float(self.params.get("sl_atr",        1.5))
        self.rr            = float(self.params.get("rr",             2.0))
        self.adx_thresh    = float(self.params.get("adx_thresh",    18.0))
        self.health_thresh = float(self.params.get("health_thresh", 70.0))

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        n = len(candles) - 1
        if n < _WARMUP:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] Warmup 1H ({n}/{_WARMUP})")

        closes_1h  = [float(c.close)  for c in candles]
        lows_1h    = [float(c.low)    for c in candles]
        highs_1h   = [float(c.high)   for c in candles]
        volumes_1h = [float(c.volume) for c in candles]

        mtf = mtf_candles or {}

        # ── 4H data ──────────────────────────────────────────────────────
        c4h = mtf.get("4h", [])
        if len(c4h) < 50:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] Waiting for 4H data ({len(c4h)} bars)")

        closes_4h = [float(c.close) for c in c4h]
        m = len(c4h) - 1

        ema20_4h = self.ema(closes_4h, 20)
        ema50_4h = self.ema(closes_4h, 50)
        e20_4h   = float(ema20_4h[m])
        e50_4h   = float(ema50_4h[m])

        ema200_4h = self.ema(closes_4h, 200) if len(c4h) >= 200 else None
        e200_4h   = float(ema200_4h[m]) if ema200_4h is not None else float('nan')

        _, bb_mid_4h, _ = self.bollinger_bands(closes_4h, 20, 2.0)
        bb_mid_v = float(bb_mid_4h[m])

        adx_arr_4h, _, _ = self.adx(c4h, 14)
        adx_v4   = float(adx_arr_4h[m])
        adx_prev = float(adx_arr_4h[m - 1]) if m >= 1 else adx_v4

        # ── 1D data ──────────────────────────────────────────────────────
        c1d = mtf.get("1d", [])
        has_1d = len(c1d) >= 55
        if has_1d:
            closes_1d = [float(c.close) for c in c1d]
            d = len(c1d) - 1
            rsi14_1d  = self.rsi(closes_1d, 14)
            ema50_1d  = self.ema(closes_1d, 50)
            e50_1d    = float(ema50_1d[d])
            rsi_1d    = float(rsi14_1d[d])
            ema200_1d = self.ema(closes_1d, 200) if len(c1d) >= 200 else None
            e200_1d   = float(ema200_1d[d]) if ema200_1d is not None else float('nan')
        else:
            e50_1d = e200_1d = rsi_1d = float('nan')

        # ── Hard exit (SELL — closes position; also prevents new entry) ──
        death_cross = (not (math.isnan(e20_4h) or math.isnan(e50_4h))
                       and e20_4h < e50_4h)

        close_below_50      = not math.isnan(e50_4h) and closes_4h[m] < e50_4h
        prev_close_below_50 = (m >= 1
                               and not math.isnan(float(ema50_4h[m - 1]))
                               and closes_4h[m - 1] < float(ema50_4h[m - 1]))
        two_bars_below = close_below_50 and prev_close_below_50

        if death_cross or two_bars_below:
            trigger = "EMA20<EMA50(4H)" if death_cross else "Close<EMA50(4H) ×2"
            return Signal(
                SignalType.SELL, self.symbol, current_price, 0,
                f"[{self.name}] Hard exit: {trigger}",
                metadata={"trigger":   trigger,
                          "ema20_4h": round(e20_4h, 2),
                          "ema50_4h": round(e50_4h, 2)},
            )

        # ── L1: Market Regime (1D) ────────────────────────────────────────
        if has_1d and not math.isnan(e50_1d):
            bull_ema = not math.isnan(e200_1d) and e50_1d > e200_1d
            bull_rsi = not math.isnan(rsi_1d)  and rsi_1d > 50.0
            if not (bull_ema or bull_rsi):
                return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                              f"[{self.name}] L1: regime bearish (RSI1D={rsi_1d:.1f})")

        # ── L2: 4H Trend ──────────────────────────────────────────────────
        if math.isnan(e20_4h) or math.isnan(e50_4h):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] L2: 4H EMA not ready")

        trend = e20_4h > e50_4h and closes_4h[m] > e20_4h
        if not math.isnan(e200_4h):
            trend = trend and e50_4h > e200_4h

        if not trend:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] L2: 4H trend not aligned")

        # ── L3: Trend Strength ────────────────────────────────────────────
        if math.isnan(adx_v4) or adx_v4 < self.adx_thresh:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] L3: ADX(4H)={adx_v4:.1f} < {self.adx_thresh}")
        if adx_v4 < adx_prev:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] L3: ADX(4H) falling ({adx_v4:.1f})")

        # ── L4: Pullback Zone ─────────────────────────────────────────────
        low_1h = lows_1h[n]
        in_pullback = (low_1h <= e20_4h or
                       (not math.isnan(bb_mid_v) and low_1h <= bb_mid_v))
        if not in_pullback:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] L4: price above pullback zone "
                          f"(low={low_1h:.2f} EMA20_4H={e20_4h:.2f})")

        # ── L5: Entry Score (1H) ───────────────────────────────────────────
        ema5  = self.ema(closes_1h, 5)
        ema9  = self.ema(closes_1h, 9)
        hma16 = self.hma(closes_1h, 16)
        rsi14 = self.rsi(closes_1h, 14)
        volma = self.sma(volumes_1h, 20)
        _, _, mh = self.macd(closes_1h, 12, 26, 9)
        roc9  = _roc(closes_1h, 9)

        e5_v   = float(ema5[n]);    e9_v  = float(ema9[n])
        hma_v  = float(hma16[n]);   rsi_v = float(rsi14[n])
        vma_v  = float(volma[n]);   vol_v = volumes_1h[n]
        mhc    = float(mh[n])
        roc9_v = float(roc9[n]) if n < len(roc9) else float('nan')
        prev_hi = highs_1h[n - 1] if n >= 1 else float('nan')

        s5 = 0
        if not (math.isnan(e5_v) or math.isnan(e9_v)) and e5_v > e9_v:         s5 += 1
        if not math.isnan(hma_v)   and closes_1h[n] > hma_v:                   s5 += 1
        if not math.isnan(rsi_v)   and rsi_v > 50.0:                            s5 += 1
        if not math.isnan(roc9_v)  and roc9_v > 0.0:                            s5 += 1
        if not math.isnan(vma_v)   and vma_v > 0 and vol_v > vma_v:            s5 += 1
        if not math.isnan(mhc)     and mhc > 0.0:                               s5 += 1
        if not math.isnan(prev_hi) and closes_1h[n] > prev_hi:                 s5 += 1

        if s5 < 6:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] L5: {s5}/7 (need 6)")

        # ── L6: Momentum Filter ───────────────────────────────────────────
        kst_arr, kst_sig = _kst(closes_1h)
        roc3 = _roc(closes_1h, 3)

        kst_v  = float(kst_arr[n]);  kst_s  = float(kst_sig[n])
        mhp    = float(mh[n - 1]) if n >= 1 else mhc
        roc3_v = float(roc3[n])     if n < len(roc3) else float('nan')
        roc3_p = float(roc3[n - 1]) if (n >= 1 and (n - 1) < len(roc3)) else float('nan')

        s6 = 0
        if not (math.isnan(kst_v) or math.isnan(kst_s)) and kst_v > kst_s:         s6 += 1
        if not (math.isnan(mhc)   or math.isnan(mhp))   and mhc > mhp:             s6 += 1
        if not (math.isnan(roc3_v) or math.isnan(roc3_p)) and roc3_v > roc3_p:     s6 += 1

        if s6 < 2:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] L6: {s6}/3 momentum (need 2)")

        # ── L7: Health Score ──────────────────────────────────────────────
        health = 0
        if not (math.isnan(e20_4h) or math.isnan(e50_4h))  and e20_4h > e50_4h:    health += 20
        if not (math.isnan(e50_4h) or math.isnan(e200_4h)) and e50_4h > e200_4h:   health += 15
        if not (math.isnan(e50_1d) or math.isnan(e200_1d)) and e50_1d > e200_1d:   health += 20
        if not math.isnan(roc9_v) and roc9_v > 0:                                    health += 10
        if not math.isnan(mhc)    and mhc > 0:                                       health += 10
        if not math.isnan(rsi_v)  and 55.0 <= rsi_v <= 75.0:                        health += 10
        if not math.isnan(vma_v)  and vma_v > 0 and vol_v > vma_v:                  health += 15

        if health < self.health_thresh:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] L7: Health={health} < {self.health_thresh:.0f}")

        # ── Entry confirmed ───────────────────────────────────────────────
        atr14 = self.atr(candles, 14)
        atr_v = float(atr14[n])
        if math.isnan(atr_v):
            atr_v = current_price * 0.015

        sl_dist = self.sl_atr * atr_v        # 1R
        sl_p    = round(current_price - sl_dist, 4)
        tp_p    = round(current_price + self.rr * sl_dist, 4)  # 2R primary TP (exchange OCO)
        tp1_1r  = round(current_price + sl_dist, 4)             # 1R reference
        tp3_trail = round(e20_4h, 4)                            # EMA20(4H) trailing level

        return Signal(
            type=SignalType.BUY,
            symbol=self.symbol,
            price=current_price,
            amount=0.0,
            confidence=min(0.60 + s5 * 0.04 + s6 * 0.03, 0.95),
            reason=(f"[{self.name}] L5={s5}/7 L6={s6}/3 "
                    f"Health={health} ADX4H={adx_v4:.1f} RSI={rsi_v:.1f}"),
            metadata={
                "stop_loss":   sl_p,
                "take_profit": tp_p,
                "tp1_1R_25pct":  tp1_1r,
                "tp3_trail_50pct": tp3_trail,
                "atr":         round(atr_v, 4),
                "sl_atr":      self.sl_atr,
                "rr":          self.rr,
                "health":      health,
                "score_l5":    s5,
                "score_l6":    s6,
                "adx_4h":      round(adx_v4, 1),
                "ema20_4h":    round(e20_4h, 2),
                "ema50_4h":    round(e50_4h, 2),
                "rsi_1h":      round(rsi_v, 1) if not math.isnan(rsi_v) else None,
            },
        )
