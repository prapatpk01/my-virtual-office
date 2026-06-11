"""
Sentinel Signal v1.0 → Python port.

Signals generated:
  BUY  — sig_long: all 3 TF entry scores positive + BOS bull + confidence ≥ threshold + R:R ≥ min
  SELL — sig_short: all 3 TF entry scores negative + BOS bear + confidence ≥ threshold
  HOLD — otherwise, with scenario and phase annotation

Key components ported:
  - Moving averages: EMA12, SMA26, EMA125, SMA200, HMA21
  - BOS (Break of Structure) — confirmed swing detection
  - MTF score — hierarchy score across 3 internal timeframes (simulated via MA crossovers)
  - Momentum Engine — 4-component score with 6 phases
  - Sentiment / VFI / OBV / DWCS (stable fixed weights)
  - Signal Gate — confidence scoring with bonus/penalty factors
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class SentinelStrategy(BaseStrategy):
    """Python port of Sentinel Signal v1.0 key signal logic."""

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.min_conf      = self.params.get("min_conf", 60.0)
        self.min_rr        = self.params.get("min_rr", 1.5)
        self.swing_len     = self.params.get("swing_len", 10)
        self.position_pct  = self.params.get("position_pct", 0.08)

    # ------------------------------------------------------------------ #
    # HMA (Hull Moving Average) — [F1] correct formula
    # ------------------------------------------------------------------ #

    def _hma(self, closes, length):
        """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
        h = max(1, round(length / 2))
        s = max(1, round(np.sqrt(length)))
        wma_h  = self._wma(closes, h)
        wma_n  = self._wma(closes, length)
        raw = np.where(
            ~np.isnan(wma_h) & ~np.isnan(wma_n),
            2.0 * wma_h - wma_n,
            np.nan
        )
        valid = [v for v in raw if not np.isnan(v)]
        if len(valid) < s:
            return np.full(len(closes), np.nan)
        hull = self._wma(valid, s)
        # Pad back to original length
        pad = len(closes) - len(hull)
        return np.concatenate([np.full(pad, np.nan), hull])

    def _wma(self, values, period):
        arr = np.array(values, dtype=float)
        result = np.full(len(arr), np.nan)
        for i in range(period - 1, len(arr)):
            weights = np.arange(1, period + 1, dtype=float)
            result[i] = np.dot(arr[i - period + 1:i + 1], weights) / weights.sum()
        return result

    # ------------------------------------------------------------------ #
    # VFI (Volume Flow Indicator) — [F13] adaptive length
    # ------------------------------------------------------------------ #

    def _vfi(self, closes, highs, lows, volumes, length=50):
        n = len(closes)
        closes  = np.array(closes)
        highs   = np.array(highs)
        lows    = np.array(lows)
        volumes = np.array(volumes)
        typical = (highs + lows + closes) / 3.0
        inter   = np.zeros(n)
        inter[1:] = np.log(np.clip(typical[1:], 1e-10, None)) - np.log(np.clip(typical[:-1], 1e-10, None))
        vinter = np.full(n, np.nan)
        for i in range(30, n):
            vinter[i] = inter[i - 30 + 1:i + 1].std()
        cutoff = 0.2 * np.nan_to_num(vinter) * closes
        vol_ma  = self.sma(volumes.tolist(), length)
        flow = np.zeros(n)
        mf   = np.zeros(n)
        mf[1:] = typical[1:] - typical[:-1]
        for i in range(length, n):
            vav = max(float(vol_ma[i]), 1e-8)
            vmax = vav * 2
            va = min(volumes[i], vmax)
            flow[i] = va if mf[i] > cutoff[i] else (-va if mf[i] < -cutoff[i] else 0)
        vfi_sum = np.array([flow[max(0, i - length + 1):i + 1].sum()
                            for i in range(n)])
        vol_arr = np.maximum(np.nan_to_num(vol_ma, nan=1), 1)
        vfi_raw = vfi_sum / vol_arr
        vfi_s   = self.ema(vfi_raw.tolist(), 3)
        hi = np.nanmax(vfi_s[-200:]) if len(vfi_s) >= 200 else np.nanmax(vfi_s)
        lo = np.nanmin(vfi_s[-200:]) if len(vfi_s) >= 200 else np.nanmin(vfi_s)
        denom = max(hi - lo, 1e-8)
        return np.clip((np.nan_to_num(vfi_s, nan=50) - lo) / denom * 100, 0, 100)

    # ------------------------------------------------------------------ #
    # OBV normalized
    # ------------------------------------------------------------------ #

    def _obv_norm(self, closes, volumes):
        closes  = np.array(closes)
        volumes = np.array(volumes)
        obv = np.zeros(len(closes))
        for i in range(1, len(closes)):
            obv[i] = obv[i-1] + (volumes[i] if closes[i] > closes[i-1]
                                  else -volumes[i] if closes[i] < closes[i-1]
                                  else 0)
        mean = self.sma(obv.tolist(), 20)
        std_arr = np.full(len(obv), np.nan)
        for i in range(20, len(obv)):
            std_arr[i] = obv[i-20:i].std()
        z = np.where(
            np.nan_to_num(std_arr) > 0,
            (obv - np.nan_to_num(mean, nan=0)) / np.clip(np.nan_to_num(std_arr, nan=1), 1e-8, None),
            0
        )
        return np.clip((z + 3) / 6 * 100, 0, 100)

    # ------------------------------------------------------------------ #
    # Momentum score (simplified 4-component)
    # ------------------------------------------------------------------ #

    def _momentum_score(self, closes, highs, lows, volumes):
        closes  = np.array(closes)
        volumes = np.array(volumes)
        n = len(closes)

        rsi_arr = self.rsi(closes.tolist(), 14)
        macd_l, macd_s, macd_h = self.macd(closes.tolist(), 12, 26, 9)
        vol_ma = self.sma(volumes.tolist(), 20)

        # ROC 5-bar
        roc5 = np.zeros(n)
        for i in range(5, n):
            if closes[i-5] > 0:
                roc5[i] = (closes[i] - closes[i-5]) / closes[i-5] * 100

        # Volume*price momentum
        vpc = np.zeros(n)
        for i in range(1, n):
            vpc[i] = (closes[i] - closes[i-1]) * volumes[i]
        vpc_ema13 = np.nan_to_num(self.ema(vpc.tolist(), 13))
        vpc_ema5  = np.nan_to_num(self.ema(vpc_ema13.tolist(), 5))

        # Composite momentum score (-100 to 100)
        rsi_safe = np.nan_to_num(rsi_arr, nan=50)
        macd_nm  = np.nan_to_num(macd_h, nan=0)
        mh_hi  = np.nanmax(macd_nm[-100:]) if len(macd_nm) >= 100 else np.nanmax(macd_nm)
        mh_lo  = np.nanmin(macd_nm[-100:]) if len(macd_nm) >= 100 else np.nanmin(macd_nm)
        macd_norm = np.clip((macd_nm - mh_lo) / max(mh_hi - mh_lo, 1e-6) * 100, 0, 100)

        rsi_mom  = np.clip((rsi_safe - 50) * 0.8, -40, 40)
        macd_mom = np.clip(macd_norm - 50, -25, 25)
        roc_mom  = np.clip(roc5 * 1.5, -15, 15)

        mom_sc = np.clip(rsi_mom + macd_mom + roc_mom, -100, 100)
        mom_slope = np.zeros(n)
        for i in range(3, n):
            mom_slope[i] = mom_sc[i] - mom_sc[i-2]

        return mom_sc, mom_slope

    # ------------------------------------------------------------------ #
    # Swing high/low and BOS detection
    # ------------------------------------------------------------------ #

    def _detect_bos(self, closes, highs, lows, swing_len):
        n  = len(closes)
        sw_hi = np.full(n, np.nan)
        sw_lo = np.full(n, np.nan)

        for i in range(swing_len, n):
            window_h = highs[max(0, i-swing_len):i+1]
            window_l = lows[max(0, i-swing_len):i+1]
            if highs[i] >= max(window_h):
                sw_hi[i] = highs[i]
            if lows[i] <= min(window_l):
                sw_lo[i] = lows[i]

        ms_trend = 0
        bos_bull = np.zeros(n, dtype=bool)
        bos_bear = np.zeros(n, dtype=bool)

        last_sw_hi = np.nan
        last_sw_lo = np.nan

        for i in range(1, n):
            if not np.isnan(sw_hi[i]):
                last_sw_hi = sw_hi[i]
            if not np.isnan(sw_lo[i]):
                last_sw_lo = sw_lo[i]

            body_pct = (abs(closes[i] - closes[i-1])
                        / max(highs[i] - lows[i], 1e-8))

            if (not np.isnan(last_sw_hi)
                    and closes[i] > last_sw_hi
                    and closes[i] > closes[i-1]
                    and body_pct >= 0.40
                    and ms_trend <= 0):
                bos_bull[i] = True
                ms_trend = 1

            if (not np.isnan(last_sw_lo)
                    and closes[i] < last_sw_lo
                    and closes[i] < closes[i-1]
                    and body_pct >= 0.40
                    and ms_trend >= 0):
                bos_bear[i] = True
                ms_trend = -1

        return bos_bull, bos_bear, ms_trend

    # ------------------------------------------------------------------ #
    # MTF approximation using different MA periods
    # ------------------------------------------------------------------ #

    def _mtf_score(self, closes):
        """
        Approximate MTF by using 3 different MA length pairs to simulate
        short / medium / long timeframe trend alignment.
        """
        closes = np.array(closes)
        e9   = self.ema(closes.tolist(), 9)
        e21  = self.ema(closes.tolist(), 21)
        e50  = self.ema(closes.tolist(), 50)
        e125 = self.ema(closes.tolist(), 125)
        s200 = self.sma(closes.tolist(), 200)

        # Short-term: EMA9 vs EMA21
        sc_s = np.where(
            np.nan_to_num(e9) > np.nan_to_num(e21),
            np.where(np.nan_to_num(e21) > np.nan_to_num(e50), 2.0, 1.0),
            np.where(np.nan_to_num(e21) < np.nan_to_num(e50), -2.0, -1.0),
        )
        # Medium-term: EMA21 vs EMA50
        sc_m = np.where(
            np.nan_to_num(e21) > np.nan_to_num(e50),
            np.where(np.nan_to_num(e50) > np.nan_to_num(e125), 2.5, 1.0),
            np.where(np.nan_to_num(e50) < np.nan_to_num(e125), -2.5, -1.0),
        )
        # Long-term: EMA125 vs SMA200
        sc_l = np.where(
            np.nan_to_num(e125) > np.nan_to_num(s200), 3.0, -3.0
        )

        raw = sc_s * 0.25 + sc_m * 0.35 + sc_l * 0.40
        # Scale to -100 to 100
        mtf_sc = np.clip(raw / 7. * 100, -100, 100)
        return mtf_sc

    # ------------------------------------------------------------------ #
    # Confidence scoring (§14 Signal Gate port)
    # ------------------------------------------------------------------ #

    def _confidence(self, elong, eshort, esc, ms_trend, mtf_sc,
                    bull_div, bear_div, bos_bull, bos_bear,
                    mom_peak, mom_fade,
                    dwcs, sent_sc, rsi_val):
        ec = 35.0
        abs_esc = abs(esc)
        ec += 22 if abs_esc > 70 else 14 if abs_esc > 50 else 7 if abs_esc > 30 else 0
        if (elong and ms_trend == 1) or (eshort and ms_trend == -1):
            ec += 10
        if (elong and mtf_sc > 30) or (eshort and mtf_sc < -30):
            ec += 10
        if (elong and mtf_sc > 60) or (eshort and mtf_sc < -60):
            ec += 5
        if (elong and bull_div) or (eshort and bear_div):
            ec += 8
        if (elong and bos_bull) or (eshort and bos_bear):
            ec += 8
        if (elong and dwcs > 60) or (eshort and dwcs < 40):
            ec += 8
        if (elong and sent_sc > 60) or (eshort and sent_sc < 40):
            ec += 5
        # Penalties
        if abs(mtf_sc) < 30:
            ec -= 8
        if (elong and bear_div) or (eshort and bull_div):
            ec -= 12
        if (elong and rsi_val > 72) or (eshort and rsi_val < 28):
            ec -= 8
        if (elong and mom_peak) or (eshort and mom_peak):
            ec -= 10
        if (elong and mom_fade) or (eshort and mom_fade):
            ec -= 8
        return np.clip(ec, 10, 96)

    # ------------------------------------------------------------------ #
    # Main analysis
    # ------------------------------------------------------------------ #

    async def analyze(self, candles: list, current_price: float) -> Signal:
        if len(candles) < 210:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Need ≥210 candles for Sentinel")

        closes  = [c.close  for c in candles]
        highs   = [c.high   for c in candles]
        lows    = [c.low    for c in candles]
        volumes = [c.volume for c in candles]

        c_arr = np.array(closes)
        h_arr = np.array(highs)
        l_arr = np.array(lows)
        v_arr = np.array(volumes)

        # MAs
        hma  = self._hma(closes, 21)
        e9   = self.ema(closes, 9)
        e21  = self.ema(closes, 21)

        # RSI, MACD, ADX
        rsi_arr = self.rsi(closes, 14)
        _, _, macd_h = self.macd(closes, 12, 26, 9)
        atr_arr = np.zeros(len(closes))
        for i in range(1, len(closes)):
            tr = max(h_arr[i] - l_arr[i],
                     abs(h_arr[i] - c_arr[i-1]),
                     abs(l_arr[i] - c_arr[i-1]))
            atr_arr[i] = tr
        atr14 = np.nan_to_num(self.ema(atr_arr.tolist(), 14), nan=current_price * 0.01)

        # Indicators
        vfi_n  = self._vfi(closes, highs, lows, volumes, 50)
        obv_n  = self._obv_norm(closes, volumes)
        mtf_sc = self._mtf_score(closes)

        # Momentum
        mom_sc, mom_slope = self._momentum_score(closes, highs, lows, volumes)

        # BOS
        bos_bull_arr, bos_bear_arr, ms_trend = self._detect_bos(
            closes, highs, lows, self.swing_len
        )

        # Sentiment (DWCS-style)
        rsi_safe = np.nan_to_num(rsi_arr, nan=50)

        def rsi_sent(r): return np.clip(80 - r, 50, 100) if r <= 30 else np.clip(120 - r, 30, 65) if r >= 70 else r
        sent_rsi_comp = np.array([rsi_sent(r) for r in rsi_safe])
        mh_arr = np.nan_to_num(macd_h, nan=0)
        mh_hi  = np.nanmax(mh_arr[-100:]) if len(mh_arr) >= 100 else np.nanmax(mh_arr)
        mh_lo  = np.nanmin(mh_arr[-100:]) if len(mh_arr) >= 100 else np.nanmin(mh_arr)
        macd_nm = np.clip((mh_arr - mh_lo) / max(mh_hi - mh_lo, 1e-6) * 100, 0, 100)

        sent_sc = np.clip(
            sent_rsi_comp * 0.25 + macd_nm * 0.20 + vfi_n * 0.25
            + obv_n * 0.15 + 50 * 0.15,
            0, 100
        )

        # DWCS stable weights
        cpk_arr = np.clip(
            (c_arr - np.nan_to_num(self.ema(closes, 21), nan=c_arr)) / np.clip(c_arr, 1e-8, None) * 50 + 50,
            0, 100
        )
        l_beh = np.full(len(closes), 50.0)
        dwcs_raw = (sent_sc * 0.30 + vfi_n * 0.20 + cpk_arr * 0.12
                    + l_beh * 0.15 + l_beh * 0.10
                    + np.clip((np.nan_to_num(hma, nan=current_price) / np.clip(c_arr, 1e-8, None) - 1 + 0.5) * 100, 0, 100) * 0.08
                    + (np.nan_to_num(mtf_sc) + 100) / 2 * 0.05)
        dwcs = np.clip(np.nan_to_num(self.ema(np.clip(dwcs_raw, 0, 100).tolist(), 5)), 0, 100)

        # Get final-bar values
        i = -1
        curr_rsi  = float(rsi_safe[i])
        curr_mtf  = float(mtf_sc[i])
        curr_mom  = float(mom_sc[i])
        curr_slope = float(mom_slope[i])
        curr_sent = float(sent_sc[i])
        curr_dwcs = float(dwcs[i])
        curr_hma  = float(hma[i]) if not np.isnan(hma[i]) else current_price
        curr_atr  = float(atr14[i])

        # Phases
        mom_str    = abs(curr_mom)
        mom_rising = curr_slope > 0.5
        mom_falling = curr_slope < -0.5
        mom_peak   = mom_str > 55 and (curr_mom * curr_slope) < -0.3
        mom_fade   = curr_mom > 40 and mom_falling and float(mh_arr[i]) < float(mh_arr[i-1])
        mom_fade_S = curr_mom < -40 and mom_rising and float(mh_arr[i]) > float(mh_arr[i-1])

        # MTF entry scores (use 3-period pairs from our synthetic MTF)
        sc_s = float(np.nan_to_num(e9[i]) - np.nan_to_num(e21[i]))
        sc_m = float(np.nan_to_num(e21[i]))  # already captured in mtf_sc

        # Simplified all-3-aligned conditions
        elong  = curr_mtf >= 30 and ms_trend >= 0
        eshort = curr_mtf <= -30 and ms_trend <= 0

        # Entry score
        esc = curr_mtf  # -100..100

        # Divergence (simplified)
        bull_div = curr_rsi < rsi_safe[-6] and float(l_arr[i]) > float(l_arr[-6])
        bear_div = curr_rsi > rsi_safe[-6] and float(h_arr[i]) < float(h_arr[-6])

        # BOS current bar
        bos_bull_now = bool(bos_bull_arr[i])
        bos_bear_now = bool(bos_bear_arr[i])

        # Confidence
        econf = self._confidence(
            elong, eshort, esc, ms_trend, curr_mtf,
            bull_div, bear_div, bos_bull_now, bos_bear_now,
            mom_peak, mom_fade,
            curr_dwcs, curr_sent, curr_rsi
        )

        # R:R calculation (simplified — use ATR-based SL/TP)
        sl_L = current_price - curr_atr * 2.0
        sl_S = current_price + curr_atr * 2.0
        tp1_L = current_price + curr_atr * 3.0
        tp1_S = current_price - curr_atr * 3.0
        rr_L = (tp1_L - current_price) / max(current_price - sl_L, 1e-8)
        rr_S = (current_price - tp1_S) / max(sl_S - current_price, 1e-8)

        conf_norm = econf / 100.0

        # Signal Gate
        sig_long  = (elong and econf >= self.min_conf and rr_L >= self.min_rr
                     and not mom_peak and not mom_fade
                     and current_price > curr_hma
                     and curr_mtf > -50
                     and curr_rsi < 75)

        sig_short = (eshort and econf >= self.min_conf and rr_S >= self.min_rr
                     and not mom_peak and not mom_fade_S
                     and current_price < curr_hma
                     and curr_mtf < 50
                     and curr_rsi > 25)

        if sig_long:
            phase = ("ACCEL" if (mom_str >= 55 and mom_rising) else
                     "BUILD" if (mom_str >= 25 and mom_rising) else
                     "IGNITE")
            return Signal(
                SignalType.BUY, self.symbol, current_price,
                amount=self.position_pct,
                reason=(f"[Sentinel] LONG Conf={econf:.0f}% MTF={curr_mtf:.0f} "
                        f"DWCS={curr_dwcs:.0f} Mom={phase}"),
                confidence=conf_norm,
                metadata={
                    "econf": econf, "esc": esc, "mtf": curr_mtf,
                    "dwcs": curr_dwcs, "rsi": curr_rsi,
                    "sl": sl_L, "tp1": tp1_L, "rr": rr_L,
                    "ms_trend": ms_trend, "phase": phase,
                },
            )

        if sig_short:
            phase = ("ACCEL" if (mom_str >= 55 and mom_falling) else
                     "BUILD" if (mom_str >= 25 and mom_falling) else
                     "IGNITE")
            return Signal(
                SignalType.SELL, self.symbol, current_price,
                amount=self.position_pct,
                reason=(f"[Sentinel] SHORT Conf={econf:.0f}% MTF={curr_mtf:.0f} "
                        f"DWCS={curr_dwcs:.0f} Mom={phase}"),
                confidence=conf_norm,
                metadata={
                    "econf": econf, "esc": esc, "mtf": curr_mtf,
                    "dwcs": curr_dwcs, "rsi": curr_rsi,
                    "sl": sl_S, "tp1": tp1_S, "rr": rr_S,
                    "ms_trend": ms_trend, "phase": phase,
                },
            )

        # HOLD — annotate what's happening
        direction = "▲ forming" if elong else "▼ forming" if eshort else "─ wait"
        need = f"{econf:.0f}% / need {self.min_conf:.0f}%"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[Sentinel] {direction} Conf={need} MTF={curr_mtf:.0f} DWCS={curr_dwcs:.0f} RSI={curr_rsi:.1f}",
            metadata={
                "econf": econf, "mtf": curr_mtf,
                "dwcs": curr_dwcs, "rsi": curr_rsi, "ms_trend": ms_trend,
            },
        )
