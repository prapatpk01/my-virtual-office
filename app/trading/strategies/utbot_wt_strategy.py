"""
UT Bot + WaveTrend MTF-Bias Strategy (15m primary TF).

Long  : WT1 crosses up from below os_level (-40)
         + UT Bot bull regime
         + MTF bias (5m/15m/1h/4h WT zones) aligned bullish
             → entry when close > HMA15 AND close > prev_open
         + If MTF neutral/opposite → also need ADX≥threshold AND ROC>0

Short : WT1 crosses down from above ob_level (+45)
         + UT Bot bear regime
         + MTF bias aligned bearish
             → entry when close < HMA15 AND close < prev_open
         + If MTF neutral/opposite → also need ADX≥threshold AND ROC<0
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class UTBotWTStrategy(BaseStrategy):

    MTF_TIMEFRAMES = ["5m", "1h", "4h"]   # fetched by bot in addition to main TF

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        # ── UT Bot ────────────────────────────────────────────────────
        self.ut_key_value  = self.params.get("ut_key_value",  1.5)
        self.ut_atr_period = self.params.get("ut_atr_period", 10)
        # ── WaveTrend ─────────────────────────────────────────────────
        self.wt_n1     = self.params.get("wt_n1",  10)
        self.wt_n2     = self.params.get("wt_n2",  21)
        self.os_level  = self.params.get("os_level", -40.0)   # oversold
        self.ob_level  = self.params.get("ob_level",  45.0)   # overbought
        # ── MTF bias neutral zone ─────────────────────────────────────
        self.bias_zone = self.params.get("bias_zone", 20.0)   # |WT1| < bias_zone = neutral
        self.bias_min  = self.params.get("bias_min",   2)     # TFs needed to agree
        # ── HMA ───────────────────────────────────────────────────────
        self.hma_period = self.params.get("hma_period", 15)
        # ── ADX ───────────────────────────────────────────────────────
        self.adx_period    = self.params.get("adx_period",    14)
        self.adx_threshold = self.params.get("adx_threshold", 20)
        # ── ROC ───────────────────────────────────────────────────────
        self.roc_period = self.params.get("roc_period", 9)
        # ── Misc ──────────────────────────────────────────────────────
        self.position_pct = self.params.get("position_pct", 0.08)

    # ─────────────────────────────────────────────────────────────── #
    # Indicators
    # ─────────────────────────────────────────────────────────────── #

    def _wma(self, prices, period: int) -> np.ndarray:
        p = np.asarray(prices, dtype=float)
        n = len(p)
        out = np.full(n, np.nan)
        if period < 1 or n < period:
            return out
        w    = np.arange(1, period + 1, dtype=float)
        wsum = w.sum()
        for i in range(period - 1, n):
            seg = p[i - period + 1: i + 1]
            if not np.any(np.isnan(seg)):
                out[i] = np.dot(w, seg) / wsum
        return out

    def _hma(self, prices, period: int) -> np.ndarray:
        p    = np.asarray(prices, dtype=float)
        half = max(period // 2, 1)
        sqr  = max(int(np.sqrt(period)), 1)
        wma_h = self._wma(p, half)
        wma_f = self._wma(p, period)
        raw   = 2.0 * wma_h - wma_f
        # Fill NaN with 0 so final WMA doesn't cascade NaN
        raw_fill = np.where(np.isnan(raw), 0.0, raw)
        result   = self._wma(raw_fill, sqr)
        # Re-mask where input was NaN
        result[np.isnan(raw)] = np.nan
        return result

    def _ut_bot(self, candles):
        """UT Bot Alert trailing stop. Returns (trail, direction, buy_cross, sell_cross)."""
        closes = np.array([c.close for c in candles], dtype=float)
        n      = len(closes)
        atr_v  = np.nan_to_num(np.array(self.atr(candles, self.ut_atr_period), dtype=float))
        nlos   = self.ut_key_value * atr_v

        trail     = np.zeros(n)
        direction = np.ones(n, dtype=int)

        for i in range(1, n):
            pc, cc = closes[i - 1], closes[i]
            pt     = trail[i - 1]
            if cc > pt:
                trail[i] = max(pt, cc - nlos[i])
            else:
                trail[i] = min(pt, cc + nlos[i])
            if direction[i - 1] == -1 and cc > pt:
                direction[i] = 1
            elif direction[i - 1] == 1 and cc < pt:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

        buy_cross  = np.zeros(n, bool)
        sell_cross = np.zeros(n, bool)
        for i in range(1, n):
            buy_cross[i]  = direction[i - 1] == -1 and direction[i] == 1
            sell_cross[i] = direction[i - 1] == 1  and direction[i] == -1
        return trail, direction, buy_cross, sell_cross

    def _wt(self, candles):
        """WaveTrend on candle list. Returns (wt1, wt2) arrays."""
        h  = np.array([c.high  for c in candles], dtype=float)
        l  = np.array([c.low   for c in candles], dtype=float)
        c  = np.array([c.close for c in candles], dtype=float)
        ap = (h + l + c) / 3.0
        esa     = np.array(self.ema(ap.tolist(), self.wt_n1), dtype=float)
        esa_f   = np.where(np.isnan(esa), ap, esa)
        d_raw   = np.array(self.ema(np.abs(ap - esa_f).tolist(), self.wt_n1), dtype=float)
        d       = np.where(np.isnan(d_raw) | (d_raw < 1e-10), 1e-10, d_raw)
        ci      = (ap - esa_f) / (0.015 * d)
        wt1     = np.nan_to_num(np.array(self.ema(ci.tolist(), self.wt_n2), dtype=float))
        wt2     = np.nan_to_num(np.array(self.sma(wt1.tolist(), 4), dtype=float))
        return wt1, wt2

    def _roc(self, closes, period: int) -> np.ndarray:
        c   = np.asarray(closes, dtype=float)
        out = np.zeros(len(c))
        for i in range(period, len(c)):
            if c[i - period] != 0:
                out[i] = (c[i] - c[i - period]) / c[i - period] * 100
        return out

    def _adx(self, highs, lows, closes, period: int = 14):
        h, l, c = (np.asarray(x, dtype=float) for x in (highs, lows, closes))
        n = len(c)
        tr = np.zeros(n); pdm = np.zeros(n); mdm = np.zeros(n)
        tr[0] = h[0] - l[0]
        for i in range(1, n):
            tr[i]  = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
            up, dn = h[i] - h[i-1], l[i-1] - l[i]
            pdm[i] = up if up > dn and up > 0 else 0.0
            mdm[i] = dn if dn > up and dn > 0 else 0.0
        sa = np.full(n, np.nan); sp = np.full(n, np.nan); sm = np.full(n, np.nan)
        if n > period:
            sa[period] = tr[1:period+1].sum()
            sp[period] = pdm[1:period+1].sum()
            sm[period] = mdm[1:period+1].sum()
            for i in range(period + 1, n):
                sa[i] = sa[i-1] - sa[i-1]/period + tr[i]
                sp[i] = sp[i-1] - sp[i-1]/period + pdm[i]
                sm[i] = sm[i-1] - sm[i-1]/period + mdm[i]
        eps = 1e-8
        pdi = np.where(sa > 0, 100*sp/np.clip(sa, eps, None), np.nan)
        mdi = np.where(sa > 0, 100*sm/np.clip(sa, eps, None), np.nan)
        ds  = np.nan_to_num(pdi) + np.nan_to_num(mdi)
        dx  = np.where(ds > 0, 100*np.abs(np.nan_to_num(pdi)-np.nan_to_num(mdi))/ds, np.nan)
        adx = np.full(n, np.nan); start = 2*period
        if n > start:
            adx[start] = np.nanmean(dx[period:start])
            for i in range(start+1, n):
                if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
                    adx[i] = (adx[i-1]*(period-1) + dx[i]) / period
        return np.nan_to_num(adx, nan=0)

    def _mtf_bias(self, candles_15m, mtf_candles: dict) -> tuple[int, int]:
        """
        WT1-based MTF bias score across 5m/15m/1h/4h.
        +1 = bullish (WT1 < -bias_zone), -1 = bearish (WT1 > +bias_zone), 0 = neutral.
        Returns (score, tfs_checked).
        """
        score = 0
        checked = 0
        sources = [("15m", candles_15m)] + [(tf, mtf_candles.get(tf, [])) for tf in ("5m", "1h", "4h")]
        for tf, bars in sources:
            needed = self.wt_n1 + self.wt_n2 + 10
            if len(bars) < needed:
                continue
            wt1, _ = self._wt(bars[-120:])
            v = float(wt1[-1])
            if v < -self.bias_zone:
                score += 1
            elif v > self.bias_zone:
                score -= 1
            checked += 1
        return score, checked

    # ─────────────────────────────────────────────────────────────── #
    # Main
    # ─────────────────────────────────────────────────────────────── #

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = max(self.wt_n1 + self.wt_n2, self.hma_period, self.ut_atr_period) + 60
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        closes = [c.close for c in candles]
        highs  = [c.high  for c in candles]
        lows   = [c.low   for c in candles]
        mtf    = mtf_candles or {}

        # ── UT Bot ───────────────────────────────────────────────────
        _, ut_dir, _, _ = self._ut_bot(candles)
        ut_bull = ut_dir[-1] == 1
        ut_bear = ut_dir[-1] == -1

        # ── WaveTrend (15m) ──────────────────────────────────────────
        wt1, wt2 = self._wt(candles)
        cw1, cw2 = float(wt1[-1]), float(wt2[-1])
        pw1, pw2 = float(wt1[-2]), float(wt2[-2])
        cross_up   = pw1 <= pw2 and cw1 > cw2
        cross_down = pw1 >= pw2 and cw1 < cw2
        wt_long  = cross_up   and pw1 < self.os_level
        wt_short = cross_down and pw1 > self.ob_level

        # ── HMA 15 ───────────────────────────────────────────────────
        hma     = self._hma(closes, self.hma_period)
        curr_hma = float(np.nan_to_num(hma[-1], nan=current_price))
        prev_hma = float(np.nan_to_num(hma[-2], nan=curr_hma))
        hma_up   = curr_hma > prev_hma
        hma_dn   = curr_hma < prev_hma
        prev_open = float(candles[-2].open) if len(candles) >= 2 else current_price
        above_hma = current_price > curr_hma
        below_hma = current_price < curr_hma
        close_above_prev_open = current_price > prev_open
        close_below_prev_open = current_price < prev_open

        # ── ADX ──────────────────────────────────────────────────────
        curr_adx = float(self._adx(highs, lows, closes, self.adx_period)[-1])
        adx_ok   = curr_adx >= self.adx_threshold

        # ── ROC ──────────────────────────────────────────────────────
        curr_roc = float(self._roc(closes, self.roc_period)[-1])
        roc_bull = curr_roc > 0
        roc_bear = curr_roc < 0

        # ── MTF Bias ─────────────────────────────────────────────────
        bias_score, tfs_checked = self._mtf_bias(candles, mtf)
        mtf_bull    = bias_score >= self.bias_min
        mtf_bear    = bias_score <= -self.bias_min
        mtf_neutral = not mtf_bull and not mtf_bear

        bias_tag = f"MTF{bias_score:+d}/{tfs_checked}TF"

        # ── Long entry ───────────────────────────────────────────────
        hma_long_ok  = above_hma and close_above_prev_open and hma_up
        if wt_long and ut_bull:
            if mtf_bull and hma_long_ok:
                return Signal(
                    SignalType.BUY, self.symbol, current_price,
                    amount=self.position_pct,
                    reason=(f"[UT-WT] WT↑ OS={pw1:.0f} | UTbull | {bias_tag} ✓ | "
                            f"HMA={curr_hma:.0f} ADX={curr_adx:.0f}"),
                    confidence=0.75,
                    metadata={"wt1": cw1, "wt2": cw2, "hma": curr_hma,
                              "adx": curr_adx, "roc": curr_roc,
                              "bias": bias_score, "ut_dir": 1},
                )
            if (mtf_neutral or mtf_bear) and adx_ok and roc_bull and hma_long_ok:
                return Signal(
                    SignalType.BUY, self.symbol, current_price,
                    amount=self.position_pct,
                    reason=(f"[UT-WT] WT↑ OS={pw1:.0f} | UTbull | {bias_tag} "
                            f"ADX={curr_adx:.0f} ROC={curr_roc:+.1f}% HMA↑"),
                    confidence=0.60,
                    metadata={"wt1": cw1, "wt2": cw2, "hma": curr_hma,
                              "adx": curr_adx, "roc": curr_roc,
                              "bias": bias_score, "ut_dir": 1},
                )

        # ── Short entry ──────────────────────────────────────────────
        hma_short_ok = below_hma and close_below_prev_open and hma_dn
        if wt_short and ut_bear:
            if mtf_bear and hma_short_ok:
                return Signal(
                    SignalType.SELL, self.symbol, current_price,
                    amount=self.position_pct,
                    reason=(f"[UT-WT] WT↓ OB={pw1:.0f} | UTbear | {bias_tag} ✓ | "
                            f"HMA={curr_hma:.0f} ADX={curr_adx:.0f}"),
                    confidence=0.75,
                    metadata={"wt1": cw1, "wt2": cw2, "hma": curr_hma,
                              "adx": curr_adx, "roc": curr_roc,
                              "bias": bias_score, "ut_dir": -1},
                )
            if (mtf_neutral or mtf_bull) and adx_ok and roc_bear and hma_short_ok:
                return Signal(
                    SignalType.SELL, self.symbol, current_price,
                    amount=self.position_pct,
                    reason=(f"[UT-WT] WT↓ OB={pw1:.0f} | UTbear | {bias_tag} "
                            f"ADX={curr_adx:.0f} ROC={curr_roc:+.1f}% HMA↓"),
                    confidence=0.60,
                    metadata={"wt1": cw1, "wt2": cw2, "hma": curr_hma,
                              "adx": curr_adx, "roc": curr_roc,
                              "bias": bias_score, "ut_dir": -1},
                )

        # ── Hold ─────────────────────────────────────────────────────
        ut_tag = "UTbull" if ut_bull else "UTbear"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            (f"[UT-WT] WT1={cw1:.1f} | {ut_tag} | {bias_tag} | "
             f"ADX={curr_adx:.0f} ROC={curr_roc:+.1f}% HMA={curr_hma:.0f}"),
            metadata={"wt1": cw1, "wt2": cw2, "hma": curr_hma,
                      "adx": curr_adx, "roc": curr_roc, "bias": bias_score},
        )
