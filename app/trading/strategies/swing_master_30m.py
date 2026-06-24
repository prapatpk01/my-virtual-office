"""
SwingMaster30m — Bidirectional swing strategy on 30m bars.

Entry TF : 30m  (mtf_candles['30m'])
Trend TF : 4H   (mtf_candles['4h'])
Macro TF : 1D   (mtf_candles['1d'])

Layer 1 (4H)  Macro Trend:
  Long  — EMA20 > EMA50  AND  Close > EMA20
  Short — EMA20 < EMA50  AND  Close < EMA20  (requires allow_short=True)

Layer 2 (4H)  Strength:   ADX14 > 18  AND  ADX rising
Layer 3 (30m) Pullback:   dist(close, EMA20(30m)) <= 0.8×ATR14  OR  RSI 40-55
Layer 4 (30m) Trigger:    3/5 conditions
  Long  — EMA5>SMA9 | HMA16↑ | MACDhist>0 | ROC9>0 | Close>PrevHigh
  Short — EMA5<SMA9 | HMA16↓ | MACDhist<0 | ROC9<0 | Close<PrevLow
Layer 5 (30m) Volume:     Volume > VolMA20
Layer 6       Health:     ≥ 70 / 100
  Long  — EMA20(4H)>EMA50 +25 | MACDhist>0 +15 | ROC>0 +15
           HigherLow +15 | VolSpike +15 | ADX Rising +15
  Short — conditions mirrored

Risk: SL=1.8×ATR(30m), TP1=1R 30%, TP2=2R 30%, Runner trailing EMA20(30m) 40%
Hard exit (long): EMA5<SMA9 OR MACDhist<0 for 2 consecutive bars → SELL
Hard exit (short): EMA5>SMA9 OR MACDhist>0 for 2 consecutive bars → BUY to close
"""
import math
from .base import BaseStrategy, Signal, SignalType

_WARMUP = 60


def _roc(closes: list, period: int) -> list:
    n = len(closes)
    out = [float('nan')] * n
    for i in range(period, n):
        prev = closes[i - period]
        if abs(prev) > 1e-10:
            out[i] = (closes[i] - prev) / prev * 100.0
    return out


class SwingMaster30m(BaseStrategy):
    """6-layer bidirectional swing on 30m. Long always; short via allow_short param."""

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.sl_atr        = float(self.params.get("sl_atr",        1.8))
        self.adx_thresh    = float(self.params.get("adx_thresh",   18.0))
        self.health_thresh = float(self.params.get("health_thresh", 70.0))
        self.allow_short   = bool(self.params.get("allow_short",   False))

        # Hard-exit state
        self._in_long:        bool = False
        self._long_exit_count: int = 0
        self._in_short:       bool = False
        self._short_exit_count: int = 0

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        mtf = mtf_candles or {}
        cp  = current_price

        # 30m primary entry TF
        c30m = mtf.get("30m") or candles
        n    = len(c30m) - 1
        if n < _WARMUP:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Warmup 30m ({n}/{_WARMUP})")

        closes_30m  = [float(c.close)  for c in c30m]
        lows_30m    = [float(c.low)    for c in c30m]
        highs_30m   = [float(c.high)   for c in c30m]
        volumes_30m = [float(c.volume) for c in c30m]

        # ── 4H trend data ─────────────────────────────────────────────────
        c4h = mtf.get("4h", [])
        if len(c4h) < 50:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Need 4H data ({len(c4h)} bars)")

        closes_4h = [float(c.close) for c in c4h]
        m = len(c4h) - 1

        ema20_4h = self.ema(closes_4h, 20)
        ema50_4h = self.ema(closes_4h, 50)
        e20_4h   = float(ema20_4h[m])
        e50_4h   = float(ema50_4h[m])

        adx_arr, _, _ = self.adx(c4h, 14)
        adx_v4   = float(adx_arr[m])
        adx_prev = float(adx_arr[m - 1]) if m >= 1 else adx_v4

        # ── 30m indicators (shared across layers) ─────────────────────────
        atr14_arr = self.atr(c30m, 14)
        atr14     = float(atr14_arr[n])
        if math.isnan(atr14):
            atr14 = cp * 0.005

        ema5_arr  = self.ema(closes_30m, 5)
        sma9_arr  = self.sma(closes_30m, 9)
        hma16_arr = self.hma(closes_30m, 16)
        rsi14_arr = self.rsi(closes_30m, 14)
        ema20_30m = self.ema(closes_30m, 20)
        volma_arr = self.sma(volumes_30m, 20)
        _, _, mh  = self.macd(closes_30m, 12, 26, 9)
        roc9      = _roc(closes_30m, 9)

        e5_v   = float(ema5_arr[n]);   s9_v  = float(sma9_arr[n])
        hma_c  = float(hma16_arr[n]);  hma_p = float(hma16_arr[n - 1]) if n >= 1 else hma_c
        rsi_v  = float(rsi14_arr[n]);  e20_30 = float(ema20_30m[n])
        vma_v  = float(volma_arr[n]);  vol_v  = volumes_30m[n]
        mhc    = float(mh[n]);         mhp    = float(mh[n - 1]) if n >= 1 else mhc
        roc9_v = float(roc9[n]) if n < len(roc9) else float('nan')
        prev_hi = highs_30m[n - 1]  if n >= 1 else float('nan')
        prev_lo = lows_30m[n - 1]   if n >= 1 else float('nan')

        # ── Hard exit check (must run before entry logic) ─────────────────
        sell_sig = self._check_hard_exit_long(e5_v, s9_v, mhc, cp)
        if sell_sig:
            return sell_sig

        if self.allow_short:
            buy_close = self._check_hard_exit_short(e5_v, s9_v, mhc, cp)
            if buy_close:
                return buy_close

        # ── Layer 1: Macro Trend (4H) ──────────────────────────────────────
        if math.isnan(e20_4h) or math.isnan(e50_4h):
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] L1: 4H EMA not ready")

        long_ok  = e20_4h > e50_4h and closes_4h[m] > e20_4h
        short_ok = e20_4h < e50_4h and closes_4h[m] < e20_4h

        if not long_ok and not (self.allow_short and short_ok):
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] L1: no 4H direction "
                          f"(EMA20={e20_4h:.2f} EMA50={e50_4h:.2f})")

        direction = "long" if long_ok else "short"

        # ── Layer 2: Trend Strength (4H) ──────────────────────────────────
        if math.isnan(adx_v4) or adx_v4 < self.adx_thresh:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] L2: ADX(4H)={adx_v4:.1f} < {self.adx_thresh}")
        if adx_v4 < adx_prev:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] L2: ADX(4H) falling ({adx_v4:.1f})")

        # ── Layer 3: Pullback Zone (30m) ───────────────────────────────────
        dist_to_ema20 = abs(cp - e20_30) if not math.isnan(e20_30) else float('inf')
        in_pullback   = (dist_to_ema20 <= atr14 * 0.8 or
                         (not math.isnan(rsi_v) and 40.0 <= rsi_v <= 55.0))
        if not in_pullback:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] L3: not in pullback "
                          f"(dist={dist_to_ema20:.2f} RSI={rsi_v:.1f})")

        # ── Layer 4: Swing Reversal Trigger (30m) 3/5 ─────────────────────
        if direction == "long":
            t = 0
            if not (math.isnan(e5_v) or math.isnan(s9_v)) and e5_v > s9_v:       t += 1
            if not (math.isnan(hma_c) or math.isnan(hma_p)) and hma_c > hma_p:   t += 1
            if not math.isnan(mhc) and mhc > 0:                                   t += 1
            if not math.isnan(roc9_v) and roc9_v > 0:                             t += 1
            if not math.isnan(prev_hi) and closes_30m[n] > prev_hi:               t += 1
        else:  # short
            t = 0
            if not (math.isnan(e5_v) or math.isnan(s9_v)) and e5_v < s9_v:       t += 1
            if not (math.isnan(hma_c) or math.isnan(hma_p)) and hma_c < hma_p:   t += 1
            if not math.isnan(mhc) and mhc < 0:                                   t += 1
            if not math.isnan(roc9_v) and roc9_v < 0:                             t += 1
            if not math.isnan(prev_lo) and closes_30m[n] < prev_lo:               t += 1

        if t < 3:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] L4: {t}/5 trigger (need 3) [{direction}]")

        # ── Layer 5: Volume ────────────────────────────────────────────────
        if not math.isnan(vma_v) and vma_v > 0 and vol_v <= vma_v:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] L5: volume < VolMA20")

        # ── Layer 6: Health Score ──────────────────────────────────────────
        health = self._health(direction, e20_4h, e50_4h, adx_v4, adx_prev,
                              mhc, roc9_v, vol_v, vma_v,
                              lows_30m, highs_30m, n)

        if health < self.health_thresh:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] L6: Health={health} < {self.health_thresh:.0f} [{direction}]")

        # ── Entry ──────────────────────────────────────────────────────────
        sl_dist = self.sl_atr * atr14
        tp2     = round(cp + 2 * sl_dist, 4) if direction == "long" else round(cp - 2 * sl_dist, 4)
        tp1     = round(cp + sl_dist, 4)      if direction == "long" else round(cp - sl_dist, 4)
        sl_p    = round(cp - sl_dist, 4)      if direction == "long" else round(cp + sl_dist, 4)
        runner  = round(e20_30, 4) if not math.isnan(e20_30) else None

        if direction == "long":
            self._in_long = True
            self._long_exit_count = 0
            sig_type = SignalType.BUY
        else:
            self._in_short = True
            self._short_exit_count = 0
            sig_type = SignalType.SELL   # bot interprets as short entry

        return Signal(
            type=sig_type,
            symbol=self.symbol,
            price=cp,
            amount=0.0,
            confidence=min(0.60 + t * 0.06 + (health - 70) * 0.005, 0.92),
            reason=(f"[{self.name}] {direction.upper()} "
                    f"L4={t}/5 Health={health} ADX={adx_v4:.1f} RSI={rsi_v:.1f}"),
            metadata={
                "direction":   direction,
                "stop_loss":   sl_p,
                "take_profit": tp2,
                "tp1_1R":      tp1,
                "tp3_trail":   runner,
                "atr":         round(atr14, 4),
                "sl_atr":      self.sl_atr,
                "score_l4":    t,
                "health":      health,
                "adx_4h":      round(adx_v4, 1),
                "rsi_30m":     round(rsi_v, 1) if not math.isnan(rsi_v) else None,
            },
        )

    # ─────────────────────────────────────────────────────────────────────
    # Hard Exit
    # ─────────────────────────────────────────────────────────────────────

    def _check_hard_exit_long(self, e5: float, s9: float,
                               mhc: float, cp: float):
        """Returns SELL signal if 2 consecutive bars show bearish exit conditions."""
        if not self._in_long:
            return None

        bearish = ((not (math.isnan(e5) or math.isnan(s9)) and e5 < s9) or
                   (not math.isnan(mhc) and mhc < 0))

        if bearish:
            self._long_exit_count += 1
        else:
            self._long_exit_count = 0

        if self._long_exit_count >= 2:
            self._in_long = False
            self._long_exit_count = 0
            return Signal(
                SignalType.SELL, self.symbol, cp, 0,
                f"[{self.name}] Hard exit long: EMA5<SMA9 or MACDhist<0 ×2",
                metadata={"trigger": "hard_exit_long"},
            )
        return None

    def _check_hard_exit_short(self, e5: float, s9: float,
                                mhc: float, cp: float):
        """Returns BUY signal to close short if 2 consecutive bars show bullish conditions."""
        if not self._in_short:
            return None

        bullish = ((not (math.isnan(e5) or math.isnan(s9)) and e5 > s9) or
                   (not math.isnan(mhc) and mhc > 0))

        if bullish:
            self._short_exit_count += 1
        else:
            self._short_exit_count = 0

        if self._short_exit_count >= 2:
            self._in_short = False
            self._short_exit_count = 0
            return Signal(
                SignalType.BUY, self.symbol, cp, 0,
                f"[{self.name}] Hard exit short: EMA5>SMA9 or MACDhist>0 ×2",
                metadata={"trigger": "hard_exit_short"},
            )
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Health Score — max 100, need ≥ 70
    # ─────────────────────────────────────────────────────────────────────

    def _health(self, direction: str,
                e20_4h: float, e50_4h: float, adx: float, adx_prev: float,
                mhc: float, roc9v: float, vol: float, vma: float,
                lows: list, highs: list, n: int) -> int:
        h = 0

        # 4H Trend (25 pts)
        if not (math.isnan(e20_4h) or math.isnan(e50_4h)):
            if (direction == "long"  and e20_4h > e50_4h) or \
               (direction == "short" and e20_4h < e50_4h):
                h += 25

        # MACD Hist (15 pts)
        if not math.isnan(mhc):
            if (direction == "long"  and mhc > 0) or \
               (direction == "short" and mhc < 0):
                h += 15

        # ROC (15 pts)
        if not math.isnan(roc9v):
            if (direction == "long"  and roc9v > 0) or \
               (direction == "short" and roc9v < 0):
                h += 15

        # Structure: Higher Low (long) / Lower High (short) — 15 pts
        if n >= 10:
            if direction == "long":
                recent = min(lows[max(0, n - 4):n + 1])
                prior  = min(lows[max(0, n - 9):n - 4])
                if recent > prior:
                    h += 15
            else:
                recent = max(highs[max(0, n - 4):n + 1])
                prior  = max(highs[max(0, n - 9):n - 4])
                if recent < prior:
                    h += 15

        # Volume Spike (15 pts)
        if not math.isnan(vma) and vma > 0 and vol > vma:
            h += 15

        # ADX Rising (15 pts)
        if not (math.isnan(adx) or math.isnan(adx_prev)) and adx > adx_prev:
            h += 15

        return h
