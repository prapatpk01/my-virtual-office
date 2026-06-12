"""
MCDX Plus v2.0 → Python port.

Signals generated:
  BUY  — Golden Cross (SMA_PC crosses above SMA_LC) OR BC (bullish confirmation) OR OS
  SELL — Death Cross (SMA_PC crosses below SMA_LC) OR OB (overbought)
  HOLD — otherwise, reports DWCS score

Key indicators ported:
  - Profit Chips (PC): normalized price position
  - Locked Chips (LC): shares locked above market
  - DWCS v6: 4-pillar composite score (Momentum, Trend, Sentiment, FundFlow)
  - BC / OS / OB: event-based alerts
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
        self.dwcs_buy   = self.params.get("dwcs_buy", 53)
        self.dwcs_sell  = self.params.get("dwcs_sell", 47)
        self.min_conf   = self.params.get("min_conf", 48)
        self.position_pct = self.params.get("position_pct", 0.08)

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

    async def analyze(self, candles: list, current_price: float) -> Signal:
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

        # RSI for OS/OB
        rsi_arr = self.rsi(closes, 14)
        curr_rsi = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 50

        conf_pct = abs(curr_dwcs - 50) / 50

        if golden_cross or dwcs_buy_signal or (curr_rsi < 30 and curr_dwcs > 45):
            reason = ("Golden Cross" if golden_cross
                      else "DWCS entered Bull zone" if dwcs_buy_signal
                      else "OS + DWCS neutral")
            return Signal(
                SignalType.BUY, self.symbol, current_price,
                amount=self.position_pct,
                reason=f"[MCDX] {reason} | DWCS={curr_dwcs:.1f} PC={curr_pc:.1f}",
                confidence=min(1.0, 0.5 + conf_pct * 0.5),
                metadata={"dwcs": curr_dwcs, "pc": curr_pc, "lc": curr_lc, "rsi": curr_rsi},
            )

        if death_cross or dwcs_sell_signal or (curr_rsi > 70 and curr_dwcs < 55):
            reason = ("Death Cross" if death_cross
                      else "DWCS entered Bear zone" if dwcs_sell_signal
                      else "OB + DWCS weakening")
            return Signal(
                SignalType.SELL, self.symbol, current_price,
                amount=self.position_pct,
                reason=f"[MCDX] {reason} | DWCS={curr_dwcs:.1f} PC={curr_pc:.1f}",
                confidence=min(1.0, 0.5 + conf_pct * 0.5),
                metadata={"dwcs": curr_dwcs, "pc": curr_pc, "lc": curr_lc, "rsi": curr_rsi},
            )

        trend = "Bull" if curr_dwcs > 55 else "Bear" if curr_dwcs < 45 else "Neutral"
        cross_state = "above" if c_sma_pc > c_sma_lc else "below"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[MCDX] DWCS={curr_dwcs:.1f} ({trend}) PC_SMA {cross_state} LC_SMA",
            metadata={"dwcs": curr_dwcs, "pc": curr_pc, "lc": curr_lc, "rsi": curr_rsi},
        )


def opens_proxy(closes, i):
    """Approximate open ≈ previous close (no OHLCV open field used here)."""
    return closes[i - 1] if i > 0 else closes[i]
