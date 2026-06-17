"""
MCDX Plus v3.1 → Python port.

Signals generated:
  BUY  — Golden Cross (SMA_PC crosses above SMA_LC)
         OR DWCS Bull zone cross/sustain + ADX trend-strength confirm
         OR WaveTrend WT1 cross above WT2 in mid-zone
         OR Pullback/Rebound: EMA bounce, RSI dip rebound, Supertrend bounce
  SELL — Death Cross (SMA_PC crosses below SMA_LC) OR DWCS Bear zone cross
  HOLD — otherwise, reports DWCS + WT scores

Key indicators ported:
  - Profit Chips (PC): normalized price position
  - Locked Chips (LC): shares locked above market
  - DWCS v6: 4-pillar composite score (Momentum, Trend, Sentiment, FundFlow)
  - WaveTrend (WT1/WT2, n1=10, n2=21): momentum oscillator, overbought/oversold
  - ADX(14): Wilder Average Directional Index — trend strength gate
  - Supertrend(7, 3.0): ATR-adaptive trend direction gate
  - OBV: On-Balance Volume slope (soft confirmation, not hard gate)
  - Pullback/Rebound engine: 4-mode secondary entry in established bull trend
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class MCDXStrategy(BaseStrategy):
    """Python port of MCDX Plus v2.0 key signal logic."""

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.nr = self.params.get("length", 100)
        self.sma_pc_len = self.params.get("sma_pc_len", 10)
        self.sma_lc_len = self.params.get("sma_lc_len", 10)
        self.dwcs_buy   = self.params.get("dwcs_buy",  53)   # lowered from 57
        self.dwcs_sell  = self.params.get("dwcs_sell", 43)
        self.min_conf   = self.params.get("min_conf",  48)
        self.rvol_min   = self.params.get("rvol_min",  0.9)  # lowered from 1.1
        self.position_pct = self.params.get("position_pct", 0.08)

        # ── Pullback / Rebound params ──────────────────────────────────────
        self.pb_ema_len  = self.params.get("pb_ema_len",  21)    # EMA level for pullback
        self.pb_tol      = self.params.get("pb_tol",    0.010)   # touch tolerance (1%)
        self.pb_dwcs_min = self.params.get("pb_dwcs_min", 52)    # min DWCS to qualify trend
        self.pb_pc_low   = self.params.get("pb_pc_low",   42)    # PC dip threshold

        # ── MTF bias (1H + 4H EMA alignment) ──────────────────────────────
        # Both timeframes must show EMA_fast > EMA_slow before a BUY fires.
        self.mtf_fast_ema = self.params.get("mtf_fast_ema", 21)
        self.mtf_slow_ema = self.params.get("mtf_slow_ema", 50)

        # ── RSI gate: only enter when RSI is in a healthy momentum zone ────
        self.rsi_min_gate = self.params.get("rsi_min_gate", 35)
        self.rsi_max_gate = self.params.get("rsi_max_gate", 75)

        # ── Price confirmation: require price > EMA(21) before BUY ────────
        self.price_ema_confirm = self.params.get("price_ema_confirm", True)

        # ── ADX trend-strength gate ────────────────────────────────────────
        # Only fire primary DWCS signals when ADX confirms a trending market.
        # ADX < threshold → ranging market → skip DWCS crosses.
        self.adx_period    = self.params.get("adx_period",    14)
        self.adx_threshold = self.params.get("adx_threshold", 18)

        # ── Supertrend(7, 3.0) trend-direction gate ───────────────────────
        # Replaces/extends MTF gate with an ATR-adaptive indicator that
        # catches reversals faster than EMA crossovers.
        self.use_supertrend   = self.params.get("use_supertrend",   True)
        self.st_period        = self.params.get("st_period",        7)
        self.st_multiplier    = self.params.get("st_multiplier",    3.0)

        # ── DWCS stability: require DWCS ≥ threshold for 2 consecutive bars
        # Prevents "fringe cross" entries where DWCS barely ticked over.
        self.dwcs_stability   = self.params.get("dwcs_stability",   True)

        # ── OBV: soft signal, no longer a hard gate ───────────────────────
        self.use_obv_confirm  = self.params.get("use_obv_confirm",  True)

        # ── WaveTrend Oscillator (LazyBear WT) ────────────────────────────
        self.use_wavetrend = self.params.get("use_wavetrend", True)
        self.wt_n1         = self.params.get("wt_n1", 10)   # channel length
        self.wt_n2         = self.params.get("wt_n2", 21)   # average length
        self.wt_ob         = self.params.get("wt_ob",  60)  # overbought level
        self.wt_os         = self.params.get("wt_os", -60)  # oversold level

        # Allow overriding the strategy name for multi-instance setups (P1/P2).
        if "name" in self.params:
            self.name = self.params["name"]

    # ------------------------------------------------------------------ #
    # Chip calculations
    # ------------------------------------------------------------------ #

    def _profit_chips(self, closes, highs, lows, nr):
        """Profit Chips: % of candles where price captured profit vs recent range."""
        arr_c = np.array(closes)
        arr_h = np.array(highs)
        arr_l = np.array(lows)
        n = len(arr_c)
        pc = np.full(n, np.nan)
        for i in range(nr - 1, n):
            lo = arr_l[i - nr + 1:i + 1].min()
            hi = arr_h[i - nr + 1:i + 1].max()
            denom = max(hi - lo, 1e-8)
            x_pr = arr_c[i] * 0.96      # MCDX12 formula
            pc[i] = np.clip((x_pr - lo) / denom * 100, 0, 100)
        return pc

    def _locked_chips(self, closes, highs, lows, nr):
        """Locked Chips: proportion locked above market."""
        arr_c = np.array(closes)
        arr_h = np.array(highs)
        arr_l = np.array(lows)
        n = len(arr_c)
        lc = np.full(n, np.nan)
        for i in range(nr - 1, n):
            lo = arr_l[i - nr + 1:i + 1].min()
            hi = arr_h[i - nr + 1:i + 1].max()
            denom = max(hi - lo, 1e-8)
            x_fc = arr_c[i] * 1.04
            fc_total = np.clip((x_fc - lo) / denom * 100, 0, 100)
            lc[i] = np.clip(100 - fc_total, 0, 100)
        return lc

    # ------------------------------------------------------------------ #
    # DWCS v6 — simplified 4-pillar composite
    # ------------------------------------------------------------------ #

    def _dwcs(self, closes, highs, lows, volumes, nr):
        closes = np.array(closes)
        highs  = np.array(highs)
        lows   = np.array(lows)
        volumes = np.array(volumes)
        n = len(closes)
        if n < max(nr, 50):
            return np.full(n, 50.0)

        # PILLAR 1 — MOMENTUM
        rsi_arr = self.rsi(closes.tolist(), 14)
        macd_l, macd_s, macd_h = self.macd(closes.tolist(), 12, 26, 9)

        # Normalize MACD histogram to 0-100
        mh = macd_h
        mh_lo = np.nanmin(mh[-100:]) if len(mh) >= 100 else np.nanmin(mh)
        mh_hi = np.nanmax(mh[-100:]) if len(mh) >= 100 else np.nanmax(mh)
        denom = max(mh_hi - mh_lo, 1e-6)
        macd_norm = np.clip((mh - mh_lo) / denom * 100, 0, 100)

        # ROC
        roc = np.full(n, 50.0)
        for i in range(7, n):
            if closes[i - 7] > 0:
                roc_val = (closes[i] - closes[i - 7]) / closes[i - 7] * 100
                roc[i] = np.clip(50 + roc_val * 5, 0, 100)

        p_momentum = np.clip(
            np.nan_to_num(rsi_arr, nan=50) * 0.40
            + np.nan_to_num(macd_norm, nan=50) * 0.35
            + roc * 0.25,
            0, 100
        )

        # PILLAR 2 — TREND (EMA-based)
        fast_ema = self.ema(closes.tolist(), 21)
        slow_ema = self.ema(closes.tolist(), 50)
        trend_sc = np.where(
            fast_ema > slow_ema,
            np.clip(60 + (fast_ema - slow_ema) / np.clip(slow_ema, 1, None) * 1000, 50, 90),
            np.clip(40 - (slow_ema - fast_ema) / np.clip(slow_ema, 1, None) * 1000, 10, 50),
        )
        p_trend = np.nan_to_num(trend_sc, nan=50)

        # PILLAR 3 — SENTIMENT (RSI + MACD composite)
        rsi_safe = np.nan_to_num(rsi_arr, nan=50)
        p_sentiment = np.clip(rsi_safe * 0.50 + np.nan_to_num(macd_norm, nan=50) * 0.50, 0, 100)

        # PILLAR 4 — FUND FLOW (volume-weighted price momentum)
        vol_ma = self.sma(volumes.tolist(), 20)
        vol_ratio = np.where(vol_ma > 0, volumes / np.clip(vol_ma, 1, None), 1.0)
        fund_raw = np.where(
            closes > self.sma(closes.tolist(), 20),
            np.clip(50 + vol_ratio * 5, 50, 90),
            np.clip(50 - vol_ratio * 5, 10, 50),
        )
        p_fundflow = np.nan_to_num(fund_raw, nan=50)

        # Combine with equal weights (simplified from adaptive)
        dwcs_raw = (p_momentum * 0.35 + p_trend * 0.30
                    + p_sentiment * 0.20 + p_fundflow * 0.15)
        # ZLEMA smoothing (approximate with EMA-5)
        dwcs_arr = self.ema(np.clip(dwcs_raw, 0, 100).tolist(), 5)
        return np.nan_to_num(dwcs_arr, nan=50)

    # ------------------------------------------------------------------ #
    # ADX — Wilder Average Directional Index
    # ------------------------------------------------------------------ #

    def _adx(self, highs, lows, closes, period=14):
        """
        Returns (adx, +DI, -DI) arrays.
        ADX > 20 = trending; ADX < 15 = ranging/weak.
        """
        h = np.array(highs, dtype=float)
        l = np.array(lows,  dtype=float)
        c = np.array(closes, dtype=float)
        n = len(c)
        tr   = np.zeros(n)
        pdm  = np.zeros(n)
        mdm  = np.zeros(n)
        tr[0] = h[0] - l[0]
        for i in range(1, n):
            tr[i]  = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
            up, dn = h[i] - h[i-1], l[i-1] - l[i]
            pdm[i] = up if up > dn and up > 0 else 0.0
            mdm[i] = dn if dn > up and dn > 0 else 0.0
        str_ = np.full(n, np.nan)
        spdm = np.full(n, np.nan)
        smdm = np.full(n, np.nan)
        if n > period:
            str_[period] = tr[1:period+1].sum()
            spdm[period] = pdm[1:period+1].sum()
            smdm[period] = mdm[1:period+1].sum()
            for i in range(period + 1, n):
                str_[i] = str_[i-1] - str_[i-1] / period + tr[i]
                spdm[i] = spdm[i-1] - spdm[i-1] / period + pdm[i]
                smdm[i] = smdm[i-1] - smdm[i-1] / period + mdm[i]
        eps = 1e-8
        pdi = np.where(str_ > 0, 100 * spdm / np.clip(str_, eps, None), np.nan)
        mdi = np.where(str_ > 0, 100 * smdm / np.clip(str_, eps, None), np.nan)
        dm_sum = np.nan_to_num(pdi) + np.nan_to_num(mdi)
        dx = np.where(dm_sum > 0,
                      100 * np.abs(np.nan_to_num(pdi) - np.nan_to_num(mdi)) / dm_sum,
                      np.nan)
        adx = np.full(n, np.nan)
        start = 2 * period
        if n > start:
            adx[start] = np.nanmean(dx[period:start])
            for i in range(start + 1, n):
                if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
                    adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        return adx, np.nan_to_num(pdi, nan=0), np.nan_to_num(mdi, nan=0)

    # ------------------------------------------------------------------ #
    # Supertrend — ATR-adaptive trend direction
    # ------------------------------------------------------------------ #

    def _supertrend(self, highs, lows, closes, period=7, multiplier=3.0):
        """
        Returns (supertrend_line, direction) arrays.
        direction = +1 when bullish (price > ST line), -1 when bearish.
        """
        h = np.array(highs,  dtype=float)
        l = np.array(lows,   dtype=float)
        c = np.array(closes, dtype=float)
        n = len(c)
        tr  = np.zeros(n)
        atr = np.zeros(n)
        tr[0] = h[0] - l[0]
        for i in range(1, n):
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
        atr[period] = np.mean(tr[:period+1])
        for i in range(period + 1, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        hl2  = (h + l) / 2.0
        bu   = hl2 + multiplier * atr
        bl   = hl2 - multiplier * atr
        fu   = np.full(n, np.nan)
        fl   = np.full(n, np.nan)
        st   = np.full(n, np.nan)
        direction = np.zeros(n, dtype=int)
        for i in range(period, n):
            fu[i] = bu[i] if i == period or bu[i] < fu[i-1] or c[i-1] > fu[i-1] else fu[i-1]
            fl[i] = bl[i] if i == period or bl[i] > fl[i-1] or c[i-1] < fl[i-1] else fl[i-1]
            if i == period:
                direction[i] = 1 if c[i] > fu[i] else -1
            else:
                if direction[i-1] == -1 and c[i] > fu[i]:
                    direction[i] = 1
                elif direction[i-1] == 1 and c[i] < fl[i]:
                    direction[i] = -1
                else:
                    direction[i] = direction[i-1]
            st[i] = fl[i] if direction[i] == 1 else fu[i]
        return st, direction

    # ------------------------------------------------------------------ #
    # OBV — On-Balance Volume
    # ------------------------------------------------------------------ #

    def _obv(self, closes, volumes):
        """Cumulative OBV: +volume on up-close bars, -volume on down-close bars."""
        c = np.array(closes,  dtype=float)
        v = np.array(volumes, dtype=float)
        n = len(c)
        out = np.zeros(n)
        for i in range(1, n):
            if c[i] > c[i-1]:
                out[i] = out[i-1] + v[i]
            elif c[i] < c[i-1]:
                out[i] = out[i-1] - v[i]
            else:
                out[i] = out[i-1]
        return out

    # ------------------------------------------------------------------ #
    # WaveTrend Oscillator (LazyBear)
    # ------------------------------------------------------------------ #

    def _wavetrend(self, highs, lows, closes, n1=10, n2=21):
        """
        WaveTrend Oscillator.
          WT1 > WT2  → bullish momentum cross
          WT1 > wt_ob → overbought (avoid new longs)
          WT1 < wt_os → oversold  (potential reversal / PB entry)
        Returns (wt1, wt2) arrays.
        """
        h    = np.array(highs,  dtype=float)
        l    = np.array(lows,   dtype=float)
        c    = np.array(closes, dtype=float)
        hlc3 = (h + l + c) / 3.0
        # Fill early NaN to prevent NaN propagation through chained EMAs
        esa_raw = np.array(self.ema(hlc3.tolist(), n1), dtype=float)
        esa = np.where(np.isnan(esa_raw), hlc3, esa_raw)
        d_raw = np.array(self.ema(np.abs(hlc3 - esa).tolist(), n1), dtype=float)
        d = np.where(np.isnan(d_raw) | (np.abs(d_raw) < 1e-8), 1e-8, d_raw)
        ci  = (hlc3 - esa) / (0.015 * d)
        wt1 = np.nan_to_num(np.array(self.ema(ci.tolist(), n2), dtype=float), nan=0.0)
        wt2 = np.nan_to_num(np.array(self.sma(wt1.tolist(), 4), dtype=float), nan=0.0)
        return wt1, wt2

    # ------------------------------------------------------------------ #
    # Pullback / Rebound engine
    # ------------------------------------------------------------------ #

    def _pullback_rebound(self, closes, highs, lows, volumes,
                          pc_arr, dwcs, rsi_arr, rvol, curr_dwcs):
        """
        Secondary BUY modes for entries during a bull-trend pullback.

        Mode 1 — EMA Pullback Bounce
          Trend intact (avg DWCS ≥ pb_dwcs_min), EMA rising,
          low touched EMA within pb_tol in last 1-3 bars,
          close back above EMA, PC recovering, RSI < 68.

        Mode 3 — RSI Dip Rebound
          Strong trend (avg DWCS ≥ 55), RSI dipped to 32-48 range,
          recovering 2 bars with RSI now < 62 (not entering near overbought),
          volume above average, price above SMA20.

        Returns (signal: bool, reason: str).
        """
        n = len(closes)
        if n < self.pb_ema_len + 15:
            return False, ""

        ema_pb  = np.nan_to_num(self.ema(closes, self.pb_ema_len), nan=closes[-1])
        sma20   = np.nan_to_num(self.sma(closes, 20),              nan=closes[-1])

        price    = closes[-1]
        curr_ema = float(ema_pb[-1])
        curr_sma = float(sma20[-1])

        rsi_safe  = np.nan_to_num(rsi_arr, nan=50.0)
        curr_rsi  = float(rsi_safe[-1])
        prev_rsi  = float(rsi_safe[-2]) if n >= 2 else curr_rsi
        prev2_rsi = float(rsi_safe[-3]) if n >= 3 else prev_rsi

        curr_pc  = float(pc_arr[-1]) if not np.isnan(pc_arr[-1]) else 50.0
        prev_pc  = float(pc_arr[-2]) if not np.isnan(pc_arr[-2]) else curr_pc
        prev2_pc = float(pc_arr[-3]) if not np.isnan(pc_arr[-3]) else prev_pc

        recent_dwcs  = float(np.mean(dwcs[-5:])) if n >= 5 else curr_dwcs
        trend_bull   = recent_dwcs >= self.pb_dwcs_min
        trend_strong = recent_dwcs >= 55
        ema_rising   = ema_pb[-1] > ema_pb[-6] if n >= 6 else False

        # ── Mode 1: EMA Pullback Bounce ───────────────────────────────────
        if trend_bull and ema_rising and price > curr_ema:
            touched = any(
                float(lows[-(k + 1)]) <= curr_ema * (1.0 + self.pb_tol)
                for k in range(1, min(4, n))
            )
            if touched and curr_pc > prev_pc and curr_rsi < 68 and rvol >= 0.8:
                dist = (price - curr_ema) / curr_ema * 100
                return True, (f"PB-EMA{self.pb_ema_len} bounce +{dist:.1f}% | "
                              f"DWCS={curr_dwcs:.1f} RSI={curr_rsi:.1f} RVOL={rvol:.1f}x")

        # ── Mode 3: RSI Dip Rebound ───────────────────────────────────────
        if trend_strong and n >= 5:
            rsi_min   = min(prev_rsi, prev2_rsi)
            rsi_dipped   = 32 < rsi_min < 48
            rsi_rising_2 = curr_rsi > prev_rsi > prev2_rsi
            above_sma    = price >= curr_sma * 0.995
            if rsi_dipped and rsi_rising_2 and above_sma and rvol >= 0.9 and curr_rsi < 62:
                return True, (f"PB-RSI rebound {prev2_rsi:.0f}→{curr_rsi:.0f} | "
                              f"DWCS={curr_dwcs:.1f} RVOL={rvol:.1f}x")

        return False, ""

    # ------------------------------------------------------------------ #
    # BC signal
    # ------------------------------------------------------------------ #

    def _bc_signal(self, closes, highs, lows, volumes, pc_arr):
        n = len(closes)
        bc = np.zeros(n, dtype=bool)
        vol_ma = self.sma(volumes.tolist(), 50)
        sma20  = self.sma(closes.tolist(), 20)
        for i in range(3, n):
            if np.isnan(vol_ma[i]) or np.isnan(sma20[i]) or np.isnan(pc_arr[i]):
                continue
            vol_all = (volumes[i] > 1.5 * max(volumes[i - 1], 1e-8)
                       and closes[i] > opens_proxy(closes, i)
                       and closes[i] > closes[i - 1])
            bc1 = (closes[i] >= sma20[i]
                   and closes[i] > closes[i - 1]
                   and vol_all)
            # BC2: PC rising two-bar divergence
            bc2 = (pc_arr[i] > pc_arr[i - 1] > pc_arr[i - 2]
                   and closes[i] > closes[i - 1])
            bc[i] = bc1 or bc2
        return bc

    # ------------------------------------------------------------------ #
    # Main analysis
    # ------------------------------------------------------------------ #

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        nr = min(self.nr, len(candles))
        if len(candles) < nr + self.sma_pc_len + 5:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        closes  = [c.close  for c in candles]
        highs   = [c.high   for c in candles]
        lows    = [c.low    for c in candles]
        volumes = [c.volume for c in candles]

        pc_arr = self._profit_chips(closes, highs, lows, nr)
        lc_arr = self._locked_chips(closes, highs, lows, nr)
        sma_pc = self.sma(list(pc_arr), self.sma_pc_len)
        sma_lc = self.sma(list(lc_arr), self.sma_lc_len)
        dwcs   = self._dwcs(closes, highs, lows, volumes, nr)

        curr_pc = float(pc_arr[-1]) if not np.isnan(pc_arr[-1]) else 50
        curr_lc = float(lc_arr[-1]) if not np.isnan(lc_arr[-1]) else 50
        curr_dwcs = float(dwcs[-1])
        c_sma_pc  = float(sma_pc[-1]) if not np.isnan(sma_pc[-1]) else 50
        p_sma_pc  = float(sma_pc[-2]) if not np.isnan(sma_pc[-2]) else 50
        c_sma_lc  = float(sma_lc[-1]) if not np.isnan(sma_lc[-1]) else 50
        p_sma_lc  = float(sma_lc[-2]) if not np.isnan(sma_lc[-2]) else 50

        # Golden Cross: SMA_PC crosses above SMA_LC
        golden_cross = p_sma_pc <= p_sma_lc and c_sma_pc > c_sma_lc
        death_cross  = p_sma_pc >= p_sma_lc and c_sma_pc < c_sma_lc

        # DWCS threshold signals
        dwcs_buy_signal  = curr_dwcs > self.dwcs_buy and dwcs[-2] <= self.dwcs_buy
        dwcs_sell_signal = curr_dwcs < self.dwcs_sell and dwcs[-2] >= self.dwcs_sell

        # RSI + RVOL for confirmation
        rsi_arr = self.rsi(closes, 14)
        curr_rsi = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 50
        vol_arr = np.array(volumes)
        vol_ma  = float(np.mean(vol_arr[-20:])) if len(vol_arr) >= 20 else float(np.mean(vol_arr))
        rvol    = float(vol_arr[-1]) / max(vol_ma, 1e-8)

        conf_pct = abs(curr_dwcs - 50) / 50

        # RSI gate: momentum zone check
        rsi_gate_ok = self.rsi_min_gate <= curr_rsi <= self.rsi_max_gate

        # Price > EMA(21) confirmation (local uptrend)
        ema21_arr = self.ema(closes, self.pb_ema_len)
        price_ok  = (not self.price_ema_confirm
                     or np.isnan(ema21_arr[-1])
                     or current_price > float(ema21_arr[-1]))

        # ── ADX trend-strength gate ───────────────────────────────────────
        adx_arr, _, _ = self._adx(highs, lows, closes, self.adx_period)
        curr_adx = float(np.nan_to_num(adx_arr[-1], nan=0))

        # ── Supertrend direction gate ─────────────────────────────────────
        st_line, st_dir = self._supertrend(highs, lows, closes,
                                           self.st_period, self.st_multiplier)
        st_bullish = bool(st_dir[-1] == 1) if self.use_supertrend else True
        curr_st    = float(np.nan_to_num(st_line[-1], nan=0))

        # ── OBV rising (soft signal — used for confidence, not hard gate) ─
        obv_raw  = self._obv(closes, volumes)
        obv_ema5 = self.ema(obv_raw.tolist(), 5)
        obv_rising = (
            len(obv_ema5) >= 3
            and not any(np.isnan([obv_ema5[-1], obv_ema5[-2], obv_ema5[-3]]))
            and float(obv_ema5[-1]) > float(obv_ema5[-2]) > float(obv_ema5[-3])
        ) if self.use_obv_confirm else True

        # ── WaveTrend ─────────────────────────────────────────────────────
        wt1_arr, wt2_arr = (self._wavetrend(highs, lows, closes, self.wt_n1, self.wt_n2)
                            if self.use_wavetrend else (np.full(len(closes), 0.0),
                                                        np.full(len(closes), 0.0)))
        curr_wt1   = float(np.nan_to_num(wt1_arr[-1], nan=0))
        curr_wt2   = float(np.nan_to_num(wt2_arr[-1], nan=0))
        prev_wt1   = float(np.nan_to_num(wt1_arr[-2], nan=0))
        prev_wt2   = float(np.nan_to_num(wt2_arr[-2], nan=0))
        wt_cross_up   = prev_wt1 <= prev_wt2 and curr_wt1 > curr_wt2   # bullish cross
        wt_overbought = curr_wt1 > self.wt_ob
        wt_oversold   = curr_wt1 < self.wt_os

        # ── MTF bias gate: 1H + 4H EMA(fast/slow) both bullish ────────────
        mtf_bias_bull = True
        mtf_label = ""
        if mtf_candles:
            blocked_by = []
            for tf in ("1h", "4h"):
                tf_bars = mtf_candles.get(tf, [])
                if len(tf_bars) >= self.mtf_slow_ema + 1:
                    tf_closes = [float(c.close) for c in tf_bars]
                    tf_fast = float(self.ema(tf_closes, self.mtf_fast_ema)[-1])
                    tf_slow = float(self.ema(tf_closes, self.mtf_slow_ema)[-1])
                    if tf_fast <= tf_slow:
                        blocked_by.append(tf)
            if blocked_by:
                mtf_bias_bull = False
                mtf_label = f" [MTF blocked: {','.join(blocked_by)} bear]"

        # ── Regime gate: MTF + Supertrend must both be bullish ───────────
        # Supertrend (ATR-adaptive) catches reversals faster than EMA cross.
        regime_blocked = not mtf_bias_bull or (self.use_supertrend and not st_bullish)
        regime_notes   = []
        if not mtf_bias_bull:
            regime_notes.append(f"MTF bear{mtf_label}")
        if self.use_supertrend and not st_bullish:
            regime_notes.append(f"Supertrend bear (ST={curr_st:.0f})")

        # ── Primary BUY: Golden Cross OR DWCS bull zone cross/sustain ────
        dwcs_stable = (curr_dwcs >= self.dwcs_buy and float(dwcs[-2]) >= self.dwcs_buy
                       if self.dwcs_stability else True)
        adx_ok      = curr_adx >= self.adx_threshold

        # OBV is a soft weight — worsens confidence rather than hard-blocking.
        # WaveTrend overbought blocks new entries to avoid chasing tops.
        wt_block = self.use_wavetrend and wt_overbought

        dwcs_buy_confirmed = (
            not wt_block and (
                # Fresh cross above dwcs_buy: ADX + RSI zone + local uptrend
                (dwcs_buy_signal and adx_ok and rsi_gate_ok and price_ok)
                or
                # Sustained above threshold: 2-bar stable + moderate ADX
                (dwcs_stable and adx_ok and rsi_gate_ok and price_ok
                 and curr_dwcs > self.dwcs_buy + 2)
                or
                # WaveTrend bullish cross in non-overbought zone — standalone entry
                (wt_cross_up and not wt_overbought and curr_dwcs >= 50
                 and adx_ok and rsi_gate_ok and price_ok)
            )
        )
        golden_cross_ok = golden_cross and price_ok

        if (golden_cross_ok or dwcs_buy_confirmed) and regime_blocked:
            return Signal(
                SignalType.HOLD, self.symbol, current_price, 0,
                f"[{self.name}] BUY blocked — {'; '.join(regime_notes)} "
                f"DWCS={curr_dwcs:.1f} WT={curr_wt1:.0f} ADX={curr_adx:.0f}",
                metadata={"dwcs": curr_dwcs, "pc": curr_pc, "lc": curr_lc,
                          "rsi": curr_rsi, "rvol": rvol,
                          "adx": curr_adx, "st_bull": st_bullish,
                          "wt1": curr_wt1, "wt2": curr_wt2},
            )
        if golden_cross_ok or dwcs_buy_confirmed:
            if golden_cross_ok:
                reason = "Golden Cross"
            elif wt_cross_up and curr_dwcs >= 50:
                reason = f"WT cross WT1={curr_wt1:.1f}→{curr_wt2:.1f} DWCS={curr_dwcs:.1f}"
            else:
                reason = f"DWCS Bull zone DWCS={curr_dwcs:.1f} ADX={curr_adx:.0f}"
            obv_note = " OBV↑" if obv_rising else ""
            return Signal(
                SignalType.BUY, self.symbol, current_price,
                amount=self.position_pct,
                reason=(f"[MCDX] {reason} | RSI={curr_rsi:.1f} RVOL={rvol:.1f}x{obv_note}"),
                confidence=min(1.0, 0.5 + conf_pct * 0.5 + (0.05 if obv_rising else 0)),
                metadata={"dwcs": curr_dwcs, "pc": curr_pc, "lc": curr_lc,
                          "rsi": curr_rsi, "rvol": rvol, "adx": curr_adx,
                          "wt1": curr_wt1, "wt2": curr_wt2, "signal": "breakout"},
            )

        # ── Primary SELL: Death Cross OR DWCS bear zone cross ────────────
        dwcs_sell_confirmed = dwcs_sell_signal and rvol >= self.rvol_min
        if death_cross or dwcs_sell_confirmed:
            reason = "Death Cross" if death_cross else f"DWCS Bear zone + RVOL={rvol:.1f}x"
            return Signal(
                SignalType.SELL, self.symbol, current_price,
                amount=self.position_pct,
                reason=f"[MCDX] {reason} | DWCS={curr_dwcs:.1f} RSI={curr_rsi:.1f}",
                confidence=min(1.0, 0.5 + conf_pct * 0.5),
                metadata={"dwcs": curr_dwcs, "pc": curr_pc, "lc": curr_lc,
                          "rsi": curr_rsi, "rvol": rvol, "signal": "breakdown"},
            )

        # ── Secondary BUY: Pullback / Rebound (fires when trend intact) ───
        pb_sig, pb_reason = self._pullback_rebound(
            closes, highs, lows, volumes, pc_arr, dwcs, rsi_arr, rvol, curr_dwcs
        )
        if pb_sig and regime_blocked:
            pb_sig = False  # regime gate suppresses all pullback entries too

        # ── PB Mode 4: Supertrend bounce (new in v3) ─────────────────────
        # Price touched the Supertrend support line and recovered.
        if (not pb_sig and self.use_supertrend and st_bullish
                and not regime_blocked and curr_adx >= self.adx_threshold
                and curr_st > 0 and len(lows) >= 3):
            touched_st = any(
                float(lows[-(k + 1)]) <= curr_st * 1.005
                for k in range(1, 3)
            )
            above_st = current_price > curr_st * 1.001
            if touched_st and above_st and rsi_gate_ok and rvol >= 0.8:
                avg_dwcs_5 = float(np.nanmean(dwcs[-5:]))
                if avg_dwcs_5 >= 52:
                    pb_sig    = True
                    pb_reason = (f"PB-ST bounce +{(current_price/curr_st-1)*100:.1f}% | "
                                 f"DWCS={curr_dwcs:.1f} ADX={curr_adx:.0f}")

        if pb_sig:
            return Signal(
                SignalType.BUY, self.symbol, current_price,
                amount=self.position_pct,
                reason=f"[MCDX] {pb_reason}",
                confidence=min(0.82, 0.42 + conf_pct * 0.40),
                metadata={"dwcs": curr_dwcs, "pc": curr_pc, "lc": curr_lc,
                          "rsi": curr_rsi, "rvol": rvol, "signal": "pullback"},
            )

        trend     = "Bull" if curr_dwcs > 55 else "Bear" if curr_dwcs < 45 else "Neutral"
        st_state  = "↑" if st_bullish else "↓"
        wt_state  = "OB" if wt_overbought else "OS" if wt_oversold else f"{curr_wt1:.0f}"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            (f"[MCDX] DWCS={curr_dwcs:.1f} ({trend}) "
             f"WT={wt_state} ADX={curr_adx:.0f} ST{st_state} RSI={curr_rsi:.0f}"),
            metadata={"dwcs": curr_dwcs, "pc": curr_pc, "lc": curr_lc,
                      "rsi": curr_rsi, "rvol": rvol,
                      "adx": curr_adx, "st_bull": st_bullish,
                      "wt1": curr_wt1, "wt2": curr_wt2},
        )


def opens_proxy(closes, i):
    """Approximate open ≈ previous close (no OHLCV open field used here)."""
    return closes[i - 1] if i > 0 else closes[i]
