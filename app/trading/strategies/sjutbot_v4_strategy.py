"""
SJ-UTBot v4 Strategy — BaseStrategy wrapper for production use.

Primary timeframe: 30m  (set tf="30m" in params)
MTF: 1h + 4h

Algorithm: UTBot trailing-stop cross on 30m is the mandatory trigger.
Fires only when ≥8 of 9 confirmations agree (HMA trend, ADX>35, swing
structure, S/R breakout, volume, 1H trend, 4H trend, EMA cross).
Risk: SL = sl_mult×ATR, TP1 = 1.2R (close tp1_fraction), TP2 = regime-aware.

Backtest Jan-May 2026: 22 trades, WR 68.2%, Net +$23.37, MaxDD -2.28%, 0 liq.
"""
import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType


# ── Internal indicators (ported from sj_utbot_v4_PRODUCTION.py) ─────────────

def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def _wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def _hma(s: pd.Series, n: int) -> pd.Series:
    half, sq = max(int(n / 2), 1), max(int(np.sqrt(n)), 1)
    return _wma(2 * _wma(s, half) - _wma(s, n), sq)


def _true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)


def _adx_di(df: pd.DataFrame, n: int):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr_rma   = _rma(_true_range(df), n)
    plus_di  = 100 * _rma(pd.Series(plus_dm,  index=df.index), n) / tr_rma
    minus_di = 100 * _rma(pd.Series(minus_dm, index=df.index), n) / tr_rma
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return plus_di, minus_di, _rma(dx, n)


def _bars_since(cond: pd.Series) -> pd.Series:
    idx = np.arange(len(cond))
    last = np.where(cond.values, idx, np.nan)
    last = pd.Series(last, index=cond.index).ffill()
    return pd.Series(idx, index=cond.index) - last


def _merge_htf(df_primary: pd.DataFrame, df_htf: pd.DataFrame, freq_htf: str, prefix: str) -> pd.Series:
    """Attach the most recent closed HTF candle close to each primary bar (no lookahead)."""
    htf = df_htf.copy()
    htf["avail_at"] = (htf.index + pd.Timedelta(freq_htf)).astype("datetime64[ns]")
    htf = htf.sort_values("avail_at")
    left = df_primary.reset_index()[["timestamp"]].copy()
    left["timestamp"] = left["timestamp"].astype("datetime64[ns]")
    right = htf.reset_index().rename(columns={"timestamp": f"{prefix}_open_time"})
    merged = pd.merge_asof(left, right, left_on="timestamp", right_on="avail_at", direction="backward")
    merged = merged.set_index("timestamp")
    return merged["close"].rename(f"{prefix}_close")


class _SJUTBotV4Core:
    """Pandas signal engine (stateless — re-computed each call)."""

    PARAMS = dict(
        ut_mult=0.30, ut_len=14,
        ema_fast=5, sma_mid=9, cross_window=3,
        hma_len=20, hma_slope_bars=2,
        adx_len=14, adx_min=35,
        swing_len=5, sr_len=15,
        vol_len=20,
        htf1_ema_len=50, htf2_ema_len=50,
        min_score=8,
        atr_len=14, sl_mult=1.0,
        tp1_r=1.2,
        bb_len=20, regime_ma_len=50,
        tp2_r_by_regime={"STRONG_TREND": 3.0, "WEAK_TREND": 2.0,
                         "BREAKOUT": 2.5, "RANGING": 1.5, "VOLATILE_CHOP": 1.5},
    )

    def __init__(self, params: dict = None):
        self.p = {**self.PARAMS, **(params or {})}

    def compute(self, df30: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame) -> pd.DataFrame:
        p = self.p
        df = df30.copy()
        n = len(df)

        ha_c = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        hc = ha_c.values
        ut_atr = _rma(_true_range(df), p["ut_len"])
        slval = (ut_atr * p["ut_mult"]).values
        tsl = np.full(n, np.nan)
        tsl[0] = hc[0] + slval[0] if not np.isnan(slval[0]) else np.nan
        for i in range(1, n):
            prev = tsl[i - 1]
            if np.isnan(prev):
                tsl[i] = hc[i] + slval[i]
            elif hc[i] > prev and hc[i - 1] > prev:
                tsl[i] = max(prev, hc[i] - slval[i])
            elif hc[i] < prev and hc[i - 1] < prev:
                tsl[i] = min(prev, hc[i] + slval[i])
            elif hc[i] > prev:
                tsl[i] = hc[i] - slval[i]
            else:
                tsl[i] = hc[i] + slval[i]
        df["tsl"] = tsl
        hcS = pd.Series(hc, index=df.index)
        df["ut_buy"]  = (hcS.shift(1) < df["tsl"].shift(1)) & (hcS > df["tsl"])
        df["ut_sell"] = (hcS.shift(1) > df["tsl"].shift(1)) & (hcS < df["tsl"])
        df["atr"] = _rma(_true_range(df), p["atr_len"])

        ema5 = df["close"].ewm(span=p["ema_fast"], adjust=False).mean()
        sma9 = df["close"].rolling(p["sma_mid"]).mean()
        crossover  = (ema5 > sma9) & (ema5.shift(1) <= sma9.shift(1))
        crossunder = (ema5 < sma9) & (ema5.shift(1) >= sma9.shift(1))
        bs_bull = _bars_since(crossover)
        bs_bear = _bars_since(crossunder)

        hma_v  = _hma(df["close"], p["hma_len"])
        hma_up = hma_v > hma_v.shift(p["hma_slope_bars"])
        hma_dn = hma_v < hma_v.shift(p["hma_slope_bars"])

        _, _, adx_v = _adx_di(df, p["adx_len"])
        swing_high = df["high"].rolling(p["swing_len"]).max()
        swing_low  = df["low"].rolling(p["swing_len"]).min()
        resistance = df["high"].rolling(p["sr_len"]).max()
        support    = df["low"].rolling(p["sr_len"]).min()
        vol_sma    = df["volume"].rolling(p["vol_len"]).mean()

        df["h1_close"] = _merge_htf(df, df1h, "1h", "h1")
        df["h4_close"] = _merge_htf(df, df4h, "4h", "h4")
        df["h1_ema"] = df["h1_close"].ewm(span=p["htf1_ema_len"], adjust=False).mean()
        df["h4_ema"] = df["h4_close"].ewm(span=p["htf2_ema_len"], adjust=False).mean()

        c2L = (ema5 > sma9) & (bs_bull <= p["cross_window"])
        c2S = (ema5 < sma9) & (bs_bear <= p["cross_window"])
        c3L = (df["close"] > hma_v) & hma_up
        c3S = (df["close"] < hma_v) & hma_dn
        c5L = df["close"] > swing_high.shift(1)
        c5S = df["close"] < swing_low.shift(1)
        c6L = df["close"] > resistance.shift(1)
        c6S = df["close"] < support.shift(1)
        c7  = df["volume"] > vol_sma
        c8L = df["h1_close"] > df["h1_ema"]
        c8S = df["h1_close"] < df["h1_ema"]
        c9L = df["h4_close"] > df["h4_ema"]
        c9S = df["h4_close"] < df["h4_ema"]

        c4 = (adx_v > p["adx_min"]).astype(int)
        df["adx"] = adx_v
        df["long_score9"]  = (1 + c2L.astype(int) + c3L.astype(int) + c5L.astype(int) +
                               c6L.astype(int) + c7.astype(int)  + c8L.astype(int) + c9L.astype(int) + c4)
        df["short_score9"] = (1 + c2S.astype(int) + c3S.astype(int) + c5S.astype(int) +
                               c6S.astype(int) + c7.astype(int)  + c8S.astype(int) + c9S.astype(int) + c4)

        # Regime (for TP2 distance only — does not gate entries)
        bb_basis  = df["close"].rolling(p["bb_len"]).mean()
        bb_std    = df["close"].rolling(p["bb_len"]).std()
        bb_width  = ((bb_basis + 2 * bb_std) - (bb_basis - 2 * bb_std)) / bb_basis * 100
        bb_w_ma   = bb_width.rolling(p["regime_ma_len"]).mean()
        atr_pct   = df["atr"] / df["close"] * 100
        atr_p_ma  = atr_pct.rolling(p["regime_ma_len"]).mean()
        is_chop   = (adx_v < 25) & (atr_pct > atr_p_ma * 1.5)
        is_strong = (adx_v > 35) & (bb_width > bb_w_ma)
        is_break  = (bb_width > bb_width.shift(3) * 1.3) & (adx_v > adx_v.shift(3))
        is_range  = (adx_v < 20) & (bb_width < bb_w_ma * 0.85)
        df["regime"] = np.select(
            [is_chop.fillna(False), is_strong.fillna(False),
             is_break.fillna(False), is_range.fillna(False)],
            ["VOLATILE_CHOP", "STRONG_TREND", "BREAKOUT", "RANGING"],
            default="WEAK_TREND",
        )
        df["tp2_r"] = df["regime"].map(p["tp2_r_by_regime"]).fillna(2.0)
        df["final_buy"]  = (df["ut_buy"]  & (df["long_score9"]  >= p["min_score"])).fillna(False)
        df["final_sell"] = (df["ut_sell"] & (df["short_score9"] >= p["min_score"])).fillna(False)
        return df


# ─── BaseStrategy wrapper ────────────────────────────────────────────────────

class SJUTBotV4Strategy(BaseStrategy):
    """
    SJUTBotV4 as a live-trading BaseStrategy.

    Primary: 30m candles (params["tf"] = "30m").
    MTF: 1h + 4h (declared in MTF_TIMEFRAMES so the bot fetches them automatically).
    """

    MTF_TIMEFRAMES = ["1h", "4h"]

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self._core = _SJUTBotV4Core(params={
            "ut_mult":     self.params.get("ut_mult",        0.30),
            "ut_len":      self.params.get("ut_len",         14),
            "adx_min":     self.params.get("adx_min",        35),
            "min_score":   self.params.get("min_score",       8),
            "atr_len":     self.params.get("atr_len",         14),
            "sl_mult":     self.params.get("sl_mult",         1.0),
            "tp1_r":       self.params.get("tp1_r",           1.2),
            "tp2_r_by_regime": self.params.get("tp2_r_by_regime", {
                "STRONG_TREND":   self.params.get("tp2_strong", 3.0),
                "WEAK_TREND":     self.params.get("tp2_weak",   2.0),
                "BREAKOUT":       self.params.get("tp2_break",  2.5),
                "RANGING":        self.params.get("tp2_range",  1.5),
                "VOLATILE_CHOP":  self.params.get("tp2_chop",   1.5),
            }),
        })
        self._sl_mult      = self.params.get("sl_mult",      1.0)
        self._tp1_r        = self.params.get("tp1_r",        1.2)
        self._tp1_fraction = self.params.get("tp1_fraction", 0.70)
        self._min_candles  = self.params.get("min_candles",  200)
        self._min_htf1     = self.params.get("min_htf1",     100)  # 1h bars
        self._min_htf2     = self.params.get("min_htf2",      50)  # 4h bars

    @staticmethod
    def _to_df(candles: list) -> pd.DataFrame:
        rows = [{"timestamp": pd.Timestamp(c.timestamp, unit="ms"),
                 "open": float(c.open),  "high": float(c.high),
                 "low":  float(c.low),   "close": float(c.close),
                 "volume": float(c.volume)} for c in candles]
        df = pd.DataFrame(rows).set_index("timestamp")
        return df

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        mtf = mtf_candles or {}
        c1h = mtf.get("1h", [])
        c4h = mtf.get("4h", [])

        if len(candles) < self._min_candles:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[SJUTBotV4] need {self._min_candles} 30m bars, have {len(candles)}")
        if len(c1h) < self._min_htf1:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[SJUTBotV4] need {self._min_htf1} 1h bars, have {len(c1h)}")
        if len(c4h) < self._min_htf2:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[SJUTBotV4] need {self._min_htf2} 4h bars, have {len(c4h)}")

        df30 = self._to_df(candles)
        df1h = self._to_df(c1h)
        df4h = self._to_df(c4h)

        try:
            computed = self._core.compute(df30, df1h, df4h)
        except Exception as e:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[SJUTBotV4] compute error: {e}")

        # candles[-1] is the still-forming bar; use [-2] = last CLOSED bar
        last = computed.iloc[-2]

        atr = float(last["atr"])
        if np.isnan(atr) or atr <= 0:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[SJUTBotV4] ATR not ready")

        regime  = str(last["regime"])
        tp2_r   = float(last["tp2_r"])
        score_l = int(last.get("long_score9",  0))
        score_s = int(last.get("short_score9", 0))

        oneR    = self._sl_mult * atr

        def _meta(side: str) -> dict:
            tp2_r_eff = tp2_r
            if side == "long":
                sl  = current_price - oneR
                tp1 = current_price + self._tp1_r * oneR
                tp2 = current_price + tp2_r_eff * oneR
            else:
                sl  = current_price + oneR
                tp1 = current_price - self._tp1_r * oneR
                tp2 = current_price - tp2_r_eff * oneR
            sl_dist = abs(current_price - sl) / current_price
            return {
                "stop_loss":    sl,
                "take_profit":  tp2,
                "sl_init":      sl,
                "tp1":          tp1,
                "tp2":          tp2,
                "atr":          atr,
                "sl_dist_pct":  sl_dist,
                "partial_pct":  self._tp1_fraction,
                "risk_pct":     0.02,
                "breakeven":    current_price,
                "rr_tp1":       self._tp1_r,
                "rr_tp2":       tp2_r_eff,
                "regime":       regime,
            }

        if bool(last["final_buy"]):
            meta = _meta("long")
            return Signal(
                SignalType.BUY, self.symbol, current_price,
                amount=0.08, reason="buy", confidence=0.82,
                metadata=meta,
            )

        if bool(last["final_sell"]):
            meta = _meta("short")
            return Signal(
                SignalType.SELL, self.symbol, current_price,
                amount=0.08, reason="sell", confidence=0.82,
                metadata=meta,
            )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            reason=(f"[SJUTBotV4] regime={regime} ADX={last['adx']:.0f} "
                    f"scoreL={score_l}/9 scoreS={score_s}/9"),
        )
