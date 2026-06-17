"""
MCDX Plus v2.1 → Python port.

Signals generated:
  BUY  — Golden Cross (SMA_PC crosses above SMA_LC)
         OR DWCS Bull zone cross + volume confirm
         OR Pullback/Rebound: EMA bounce, PC rebound, RSI dip rebound
  SELL — Death Cross (SMA_PC crosses below SMA_LC) OR DWCS Bear zone cross
  HOLD — otherwise, reports DWCS score

Key indicators ported:
  - Profit Chips (PC): normalized price position
  - Locked Chips (LC): shares locked above market
  - DWCS v6: 4-pillar composite score (Momentum, Trend, Sentiment, FundFlow)
  - Pullback/Rebound engine: 3-mode secondary entry in established bull trend
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
        self.dwcs_buy   = self.params.get("dwcs_buy",  57)
        self.dwcs_sell  = self.params.get("dwcs_sell", 43)
        self.min_conf   = self.params.get("min_conf",  48)
        self.rvol_min   = self.params.get("rvol_min",  1.1)  # volume confirm for DWCS entry
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
        # Avoids buying deeply oversold (below rsi_min) or overbought (above rsi_max).
        self.rsi_min_gate = self.params.get("rsi_min_gate", 40)
        self.rsi_max_gate = self.params.get("rsi_max_gate", 72)

        # ── Price confirmation: require price > EMA(21) before BUY ────────
        self.price_ema_confirm = self.params.get("price_ema_confirm", True)

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
    # Pullback / Rebound engine
    # ------------------------------------------------------------------ #

    def _pullback_rebound(self, closes, highs, lows, volumes,
                          pc_arr, dwcs, rsi_arr, rvol, curr_dwcs):
        """
        Three secondary BUY modes for entries during a bull-trend pullback.

        Mode 1 — EMA Pullback Bounce
          Trend intact (avg DWCS ≥ pb_dwcs_min), EMA rising,
          low touched EMA within pb_tol in last 1-3 bars,
          close back above EMA, PC recovering, RSI < 68.

        Mode 2 — PC Rebound
          PC was strong (>55) in last 15 bars, pulled back below pb_pc_low,
          now rising 2 consecutive bars, price near/above SMA20, RSI 32-65.

        Mode 3 — RSI Dip Rebound
          Strong trend (avg DWCS ≥ 55), RSI dipped to 32-48 range,
          recovering 2 bars, volume above average, price above SMA20.

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
            if rsi_dipped and rsi_rising_2 and above_sma and rvol >= 0.9:
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

        # ── MTF bias gate: 1H + 4H EMA(fast/slow) both bullish ────────────
        # Blocks BUY signals when higher-TF trend is not aligned.
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

        # ── Primary BUY: Golden Cross OR DWCS bull zone cross ─────────────
        # dwcs_buy_confirmed requires: DWCS cross + volume + RSI gate + price > EMA21
        dwcs_buy_confirmed = (dwcs_buy_signal and rvol >= self.rvol_min
                              and rsi_gate_ok and price_ok)
        # golden cross: looser (no RSI gate needed — it's a chip-distribution signal)
        golden_cross_ok = golden_cross and price_ok
        if (golden_cross_ok or dwcs_buy_confirmed) and not mtf_bias_bull:
            # Signal exists but MTF trend is against it — hold
            return Signal(
                SignalType.HOLD, self.symbol, current_price, 0,
                f"[{self.name}] BUY blocked — MTF bias bear{mtf_label} "
                f"DWCS={curr_dwcs:.1f} RVOL={rvol:.1f}x",
                metadata={"dwcs": curr_dwcs, "pc": curr_pc, "lc": curr_lc,
                          "rsi": curr_rsi, "rvol": rvol, "mtf_bias": "bear"},
            )
        if golden_cross_ok or dwcs_buy_confirmed:
            reason = "Golden Cross" if golden_cross_ok else f"DWCS Bull zone + RVOL={rvol:.1f}x"
            return Signal(
                SignalType.BUY, self.symbol, current_price,
                amount=self.position_pct,
                reason=f"[MCDX] {reason} | DWCS={curr_dwcs:.1f} RSI={curr_rsi:.1f}",
                confidence=min(1.0, 0.5 + conf_pct * 0.5),
                metadata={"dwcs": curr_dwcs, "pc": curr_pc, "lc": curr_lc,
                          "rsi": curr_rsi, "rvol": rvol, "signal": "breakout"},
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
        if pb_sig and not mtf_bias_bull:
            pb_sig = False  # MTF bear bias suppresses pullback entries too
        if pb_sig:
            return Signal(
                SignalType.BUY, self.symbol, current_price,
                amount=self.position_pct,
                reason=f"[MCDX] {pb_reason}",
                confidence=min(0.82, 0.42 + conf_pct * 0.40),
                metadata={"dwcs": curr_dwcs, "pc": curr_pc, "lc": curr_lc,
                          "rsi": curr_rsi, "rvol": rvol, "signal": "pullback"},
            )

        trend = "Bull" if curr_dwcs > 55 else "Bear" if curr_dwcs < 45 else "Neutral"
        cross_state = "above" if c_sma_pc > c_sma_lc else "below"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[MCDX] DWCS={curr_dwcs:.1f} ({trend}) PC_SMA {cross_state} LC_SMA RVOL={rvol:.1f}x",
            metadata={"dwcs": curr_dwcs, "pc": curr_pc, "lc": curr_lc,
                      "rsi": curr_rsi, "rvol": rvol},
        )


def opens_proxy(closes, i):
    """Approximate open ≈ previous close (no OHLCV open field used here)."""
    return closes[i - 1] if i > 0 else closes[i]
