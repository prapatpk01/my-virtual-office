"""
SJ-UTBot v3.1 Strategy — Adaptive Long/Short with Multi-Filter Confirmation.

LONG : UT Buy trigger + EMA5/SMA9 bull cross (within syncWindow bars)
       AND score >= filter_threshold (4 components: HMA, ADX, SwingHigh, Resistance)
SHORT: UT Sell trigger only — minimal filter for aggressive hedging

Key design: v2 is symmetric (same filter for both sides).
v3.1 is ASYMMETRIC: rigorous longs, fast shorts → ideal for futures hedge mode.

SL = sl_mult × ATR(atr_len)
TP = SL × rr
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class SJUTBotV3Strategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        # UT Bot TSL
        self.ut_mult      = self.params.get("ut_mult",          0.30)
        self.ut_len       = self.params.get("ut_len",           14)
        # Confirmation filters (long only)
        self.filter_thr   = self.params.get("filter_threshold",  3)   # need 3/4 score
        self.adx_min      = self.params.get("adx_min",          20)
        self.hma_len      = self.params.get("hma_len",          20)
        self.sync_window  = self.params.get("sync_window",       5)   # bars for EMA-UT sync
        self.swing_len    = self.params.get("swing_len",         5)   # swing high/low lookback
        self.resist_len   = self.params.get("resist_len",        15)  # resistance lookback
        # SL / TP
        self.atr_len      = self.params.get("atr_len",          14)
        self.sl_mult      = self.params.get("sl_mult",          2.5)
        self.rr           = self.params.get("rr",               1.2)
        # Direction
        self.allow_long   = self.params.get("allow_long",       True)
        self.allow_short  = self.params.get("allow_short",      True)
        self.position_pct = self.params.get("position_pct",     0.08)

    # ── Indicators ────────────────────────────────────────────────────────────

    def _wilder_atr(self, highs, lows, closes, period: int) -> np.ndarray:
        h, l, c = (np.asarray(x, float) for x in (highs, lows, closes))
        n = len(c)
        tr = np.zeros(n)
        tr[0] = h[0] - l[0]
        for i in range(1, n):
            tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        atr = np.zeros(n)
        if n >= period:
            atr[period-1] = tr[:period].mean()
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr

    def _ha(self, candles):
        """Full Heikin-Ashi: returns (ha_close, ha_open)."""
        o = np.array([c.open  for c in candles], float)
        h = np.array([c.high  for c in candles], float)
        l = np.array([c.low   for c in candles], float)
        c = np.array([c.close for c in candles], float)
        ha_c = (o + h + l + c) / 4.0
        ha_o = np.zeros(len(o))
        ha_o[0] = (o[0] + c[0]) / 2.0
        for i in range(1, len(o)):
            ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2.0
        return ha_c, ha_o

    def _tsl(self, ha_c: np.ndarray, ut_atr: np.ndarray) -> np.ndarray:
        """ATR trailing stop (Pine Script v2 logic)."""
        n = len(ha_c)
        sv = ut_atr * self.ut_mult
        tsl = np.zeros(n)
        tsl[0] = ha_c[0] + sv[0]
        for i in range(1, n):
            src, sp, pt, s = ha_c[i], ha_c[i-1], tsl[i-1], sv[i]
            if src > pt and sp > pt:
                tsl[i] = max(pt, src - s)
            elif src < pt and sp < pt:
                tsl[i] = min(pt, src + s)
            elif src > pt:
                tsl[i] = src - s
            else:
                tsl[i] = src + s
        return tsl

    def _ema(self, arr: np.ndarray, period: int) -> np.ndarray:
        k = 2.0 / (period + 1)
        out = np.empty(len(arr))
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = arr[i] * k + out[i-1] * (1 - k)
        return out

    def _sma(self, arr: np.ndarray, period: int) -> np.ndarray:
        out = np.full(len(arr), np.nan)
        for i in range(period - 1, len(arr)):
            out[i] = arr[i-period+1:i+1].mean()
        return out

    def _wma(self, arr: np.ndarray, period: int) -> np.ndarray:
        w = np.arange(1, period + 1, dtype=float)
        out = np.full(len(arr), np.nan)
        for i in range(period - 1, len(arr)):
            out[i] = np.dot(arr[i-period+1:i+1], w) / w.sum()
        return out

    def _hma(self, arr: np.ndarray, period: int) -> np.ndarray:
        half     = max(period // 2, 1)
        sqrt_p   = max(int(np.sqrt(period)), 1)
        raw      = np.nan_to_num(arr, nan=arr[~np.isnan(arr)][0] if np.any(~np.isnan(arr)) else 0)
        raw_diff = 2 * self._wma(raw, half) - self._wma(raw, period)
        return self._wma(raw_diff, sqrt_p)

    def _adx(self, highs, lows, closes, period: int) -> np.ndarray:
        h, l, c = (np.asarray(x, float) for x in (highs, lows, closes))
        n = len(c)
        tr = np.zeros(n); pdm = np.zeros(n); mdm = np.zeros(n)
        tr[0] = h[0] - l[0]
        for i in range(1, n):
            tr[i]  = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
            up, dn = h[i]-h[i-1], l[i-1]-l[i]
            if up > dn and up > 0: pdm[i] = up
            if dn > up and dn > 0: mdm[i] = dn
        sa = sp = sm = np.zeros(n), np.zeros(n), np.zeros(n)
        sa, sp, sm = np.zeros(n), np.zeros(n), np.zeros(n)
        if n > period:
            sa[period] = tr[1:period+1].sum()
            sp[period] = pdm[1:period+1].sum()
            sm[period] = mdm[1:period+1].sum()
            for i in range(period+1, n):
                sa[i] = sa[i-1] - sa[i-1]/period + tr[i]
                sp[i] = sp[i-1] - sp[i-1]/period + pdm[i]
                sm[i] = sm[i-1] - sm[i-1]/period + mdm[i]
        eps = 1e-8
        pdi = np.where(sa > 0, 100*sp/np.clip(sa, eps, None), np.nan)
        mdi = np.where(sa > 0, 100*sm/np.clip(sa, eps, None), np.nan)
        ds  = np.nan_to_num(pdi) + np.nan_to_num(mdi)
        dx  = np.where(ds > 0, 100*np.abs(np.nan_to_num(pdi)-np.nan_to_num(mdi))/np.clip(ds, eps, None), np.nan)
        adx = np.full(n, np.nan)
        start = 2*period
        if n > start:
            adx[start] = np.nanmean(dx[period:start])
            for i in range(start+1, n):
                if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
                    adx[i] = (adx[i-1]*(period-1) + dx[i]) / period
        return np.nan_to_num(adx, nan=0)

    # ── Main analysis ─────────────────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = max(self.ut_len, self.atr_len, self.resist_len,
                      self.hma_len, self.sync_window) + 40
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        h  = np.array([c.high  for c in candles], float)
        l  = np.array([c.low   for c in candles], float)
        cl = np.array([c.close for c in candles], float)
        n  = len(cl)

        # ── HA + TSL ──────────────────────────────────────────────────────────
        ut_atr = self._wilder_atr(h, l, cl, self.ut_len)
        sl_atr = self._wilder_atr(h, l, cl, self.atr_len)
        ha_c, _ = self._ha(candles)
        tsl    = self._tsl(ha_c, ut_atr)

        # UT crossover arrays (shifted for crossover: compare i-1 vs i)
        ut_buy  = np.concatenate([[False], (ha_c[:-1] < tsl[:-1]) & (ha_c[1:] > tsl[1:])])
        ut_sell = np.concatenate([[False], (ha_c[:-1] > tsl[:-1]) & (ha_c[1:] < tsl[1:])])

        # ── EMA5 / SMA9 cross ─────────────────────────────────────────────────
        ema5 = self._ema(cl, 5)
        sma9 = self._sma(cl, 9)
        ema_bull = np.concatenate([[False], (ema5[:-1] <= sma9[:-1]) & (ema5[1:] > sma9[1:])])
        # (ema_bear not needed for entry — used only for future TSL-exit logic)

        # ── Long sync window ──────────────────────────────────────────────────
        # Fire when UT buy AND EMA bull cross have BOTH occurred within syncWindow bars
        sw = self.sync_window
        recent_ut_buy   = np.any(ut_buy[max(0, n-1-sw):n-1])
        recent_ema_bull = np.any(ema_bull[max(0, n-1-sw):n-1])
        long_sync = (bool(ut_buy[-1]) and recent_ema_bull) or \
                    (bool(ema_bull[-1]) and recent_ut_buy)

        # ── Filters ───────────────────────────────────────────────────────────
        hma_arr  = self._hma(cl, self.hma_len)
        adx_arr  = self._adx(h, l, cl, 14)

        # Swing high/low and resistance/support (rolling max/min with 1-bar lag)
        swing_h  = np.array([np.max(h[max(0,i-self.swing_len+1):i+1])  for i in range(n)])
        resist   = np.array([np.max(h[max(0,i-self.resist_len+1):i+1]) for i in range(n)])

        curr_hma  = float(np.nan_to_num(hma_arr[-1], nan=current_price))
        curr_adx  = float(adx_arr[-1])
        curr_atr  = float(sl_atr[-1]) if sl_atr[-1] > 0 else current_price * 0.005
        tsl_val   = float(tsl[-1])
        tsl_green = bool(ha_c[-1] > tsl[-1])

        # 4-component long score
        score_hma    = 1 if current_price > curr_hma else 0
        score_adx    = 1 if curr_adx > self.adx_min else 0
        score_swing  = 1 if (n >= 2 and current_price > float(swing_h[-2])) else 0
        score_resist = 1 if (n >= 2 and current_price > float(resist[-2]))  else 0
        long_score   = score_hma + score_adx + score_swing + score_resist

        # ── Entry decisions ───────────────────────────────────────────────────
        final_buy  = long_sync and long_score >= self.filter_thr and self.allow_long
        final_sell = bool(ut_sell[-1]) and self.allow_short

        # SL / TP (ATR-based, different for long vs short)
        sl_long  = round(current_price - self.sl_mult * curr_atr, 2)
        tp_long  = round(current_price + self.sl_mult * self.rr * curr_atr, 2)
        sl_short = round(current_price + self.sl_mult * curr_atr, 2)
        tp_short = round(current_price - self.sl_mult * self.rr * curr_atr, 2)

        if final_buy:
            conf = min(0.90, 0.65 + 0.05 * (long_score - self.filter_thr))
            return Signal(
                SignalType.BUY, self.symbol, current_price,
                amount=self.position_pct,
                reason=(f"[SJv3] UT↑+EMA sync  score={long_score}/4  "
                        f"TSL={tsl_val:.0f}  ADX={curr_adx:.0f}"),
                confidence=conf,
                metadata={"stop_loss": sl_long, "take_profit": tp_long,
                          "tsl": tsl_val, "atr": curr_atr, "score": long_score},
            )

        if final_sell:
            return Signal(
                SignalType.SELL, self.symbol, current_price,
                amount=self.position_pct,
                reason=(f"[SJv3] UT↓ short  TSL={'🟢' if tsl_green else '🔴'}  "
                        f"TSL={tsl_val:.0f}  ADX={curr_adx:.0f}"),
                confidence=0.65,
                metadata={"stop_loss": sl_short, "take_profit": tp_short,
                          "tsl": tsl_val, "atr": curr_atr},
            )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            (f"[SJv3] TSL={'🟢' if tsl_green else '🔴'}  "
             f"score={long_score}/4  ADX={curr_adx:.0f}  sync={long_sync}"),
            metadata={"tsl": tsl_val, "atr": curr_atr,
                      "score": long_score, "tsl_green": tsl_green},
        )
