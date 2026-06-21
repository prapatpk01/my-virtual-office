"""
TrendContinuation Improved — 15m primary + 1h/4h MTF.

Primary: 15m candles. MTF: 1h + 4h.
Key change: ADX(14, 15m) > 30 gate. Re-tuned exit: TP1=0.5R (40%), SL→BE, TP2=2.5R (60%).

Backtest Jan-May 2026 ($50 / 20x / 0.04%/side): 142 trades, WR 77.5%, PnL +$212.94, MaxDD -6.07%
At OKX 0.05%/side (~0.20% RT):  ~$42.52 estimated PnL.
"""
import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType


# ── Indicators ────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def _true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)


def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _rma(_true_range(df), period)


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr_rma   = _rma(_true_range(df), n)
    plus_di  = 100 * _rma(pd.Series(plus_dm,  index=df.index), n) / tr_rma
    minus_di = 100 * _rma(pd.Series(minus_dm, index=df.index), n) / tr_rma
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return _rma(dx, n)


def _mtf_bias(primary_close: pd.Series, aligned: dict,
              ema_fast: int = 20, ema_slow: int = 50,
              rsi_period: int = 14, rsi_bull: float = 55.0, rsi_bear: float = 45.0) -> pd.Series:
    """Composite MTF bias −100…+100."""
    def _vote(s: pd.Series) -> pd.Series:
        ef, es, r = _ema(s, ema_fast), _ema(s, ema_slow), _rsi(s, rsi_period)
        v = (np.where(s > ef, 1, -1) + np.where(ef > es, 1, -1) +
             np.where(r > rsi_bull, 1, np.where(r < rsi_bear, -1, 0)))
        return pd.Series(v, index=s.index)
    votes = [_vote(primary_close)] + [_vote(c) for c in aligned.values()]
    return (sum(votes) / len(votes)) / 3.0 * 100.0


def _merge_htf(df_primary: pd.DataFrame, df_htf: pd.DataFrame,
               freq_htf: str, prefix: str) -> pd.DataFrame:
    """No-lookahead merge: each primary bar gets the last CLOSED HTF bar."""
    htf = df_htf.copy()
    htf["avail_at"] = (htf.index + pd.Timedelta(freq_htf)).astype("datetime64[ns]")
    htf = htf.sort_values("avail_at")
    left = df_primary.reset_index()[["timestamp"]].copy()
    left["timestamp"] = left["timestamp"].astype("datetime64[ns]")
    right = htf.reset_index().rename(columns={"timestamp": f"{prefix}_open_time"})
    m = pd.merge_asof(left, right, left_on="timestamp", right_on="avail_at", direction="backward")
    m = m.set_index("timestamp")
    return m[["open", "high", "low", "close", "volume"]].add_prefix(f"{prefix}_")


# ── Core computation ─────────────────────────────────────────────────────────

def _compute(df15: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame, p: dict) -> pd.DataFrame:
    out = df15.copy()
    out = out.join(_merge_htf(df15, df1h, "1h", "h1"))
    out = out.join(_merge_htf(df15, df4h, "4h", "h4"))

    ef4, es4 = _ema(out["h4_close"], p["ema_fast"]), _ema(out["h4_close"], p["ema_slow"])
    macro_up, macro_dn = ef4 > es4, ef4 < es4

    ef1, es1 = _ema(out["h1_close"], p["ema_fast"]), _ema(out["h1_close"], p["ema_slow"])
    mid_up, mid_dn = ef1 > es1, ef1 < es1

    ema20_1h = ef1
    near_ema       = (out["h1_close"] - ema20_1h).abs() / ema20_1h <= p["pullback_pct"]
    wick_bounce_l  = (out["h1_low"]  <= ema20_1h * 1.003) & (out["h1_close"] > ema20_1h)
    wick_reject_s  = (out["h1_high"] >= ema20_1h * 0.997) & (out["h1_close"] < ema20_1h)
    at_pull_long   = near_ema | wick_bounce_l
    at_pull_short  = near_ema | wick_reject_s

    comp_pct = _mtf_bias(out["close"], {"1h": out["h1_close"], "4h": out["h4_close"]})
    long_bias_ok   = comp_pct > p["bias_gate"]
    short_bias_ok  = comp_pct < -p["bias_gate"]

    ema9  = _ema(out["close"], p["ema_micro"])
    ema20 = _ema(out["close"], p["ema_fast"])
    rsi15 = _rsi(out["close"], p["rsi_period"])
    volma = _sma(out["volume"], p["vol_period"])
    out["atr"]   = _atr(out, p["atr_period"])
    out["adx15"] = _adx(out, p["adx_len"])
    vol_ok = (volma > 0) & (out["volume"] >= volma * p["vol_mult"])
    adx_ok = out["adx15"] > p["adx_min"]

    c1L = out["close"] > ema9
    c2L = out["close"] > ema20
    c3L = (rsi15 >= p["rsi_min_buy"]) & (rsi15 <= p["rsi_max_buy"])
    met_long = c1L.astype(int) + c2L.astype(int) + c3L.astype(int) + vol_ok.astype(int)

    c1S = out["close"] < ema9
    c2S = out["close"] < ema20
    c3S = (rsi15 >= p["rsi_min_sell"]) & (rsi15 <= p["rsi_max_sell"])
    met_short = c1S.astype(int) + c2S.astype(int) + c3S.astype(int) + vol_ok.astype(int)

    out["final_buy"]  = (macro_up & mid_up  & at_pull_long  & long_bias_ok  &
                          (met_long  >= p["min_entry_cond"]) & adx_ok).fillna(False)
    out["final_sell"] = (macro_dn  & mid_dn  & at_pull_short & short_bias_ok &
                          (met_short >= p["min_entry_cond"]) & adx_ok).fillna(False)

    dist = (_atr(out, p["atr_period"]) * p["sl_mult"]).clip(
        lower=out["close"] * p["sl_min_pct"],
        upper=out["close"] * p["sl_max_pct"],
    )
    out["dist"] = dist
    return out


# ── BaseStrategy wrapper ─────────────────────────────────────────────────────

class TrendContImprovedStrategy(BaseStrategy):
    """
    TrendContinuation Improved: 15m primary + 1h/4h MTF, ADX(15m)>30 filter.
    Partial-close: TP1=0.5R (close 40%), SL→BE, TP2=2.5R (60% runner).
    """

    MTF_TIMEFRAMES = ["1h", "4h"]

    DEFAULTS = dict(
        ema_fast=20, ema_slow=50, ema_micro=9, rsi_period=14,
        rsi_min_buy=35.0, rsi_max_buy=75.0,
        rsi_min_sell=28.0, rsi_max_sell=65.0,
        bias_gate=70.0,
        pullback_pct=0.010,
        vol_period=20, vol_mult=1.0, min_entry_cond=4,
        atr_period=14, sl_mult=1.2, sl_min_pct=0.012, sl_max_pct=0.035,
        adx_len=14, adx_min=30,
        tp1_r=0.5, tp1_fraction=0.40, tp2_r=2.5,
    )

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self._p = {**self.DEFAULTS, **{k: self.params.get(k, v) for k, v in self.DEFAULTS.items()}}
        self._min_primary = self.params.get("min_primary", 100)
        self._min_1h      = self.params.get("min_1h",       60)
        self._min_4h      = self.params.get("min_4h",       55)

    @staticmethod
    def _to_df(candles: list) -> pd.DataFrame:
        rows = [
            {"timestamp": pd.Timestamp(c.timestamp, unit="ms"),
             "open": float(c.open), "high": float(c.high),
             "low": float(c.low),   "close": float(c.close),
             "volume": float(c.volume)}
            for c in candles
        ]
        df = pd.DataFrame(rows).set_index("timestamp")
        return df

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        mtf = mtf_candles or {}
        c1h = mtf.get("1h", [])
        c4h = mtf.get("4h", [])

        if len(candles) < self._min_primary:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[TCImproved] need {self._min_primary} 15m bars, have {len(candles)}")
        if len(c1h) < self._min_1h:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[TCImproved] need {self._min_1h} 1h bars, have {len(c1h)}")
        if len(c4h) < self._min_4h:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[TCImproved] need {self._min_4h} 4h bars, have {len(c4h)}")

        df15 = self._to_df(candles)
        df1h = self._to_df(c1h)
        df4h = self._to_df(c4h)

        try:
            computed = _compute(df15, df1h, df4h, self._p)
        except Exception as e:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[TCImproved] compute error: {e}")

        # candles[-1] is the still-forming bar; [-2] is the last CLOSED bar
        last = computed.iloc[-2]

        buy  = bool(last["final_buy"])
        sell = bool(last["final_sell"])
        if not buy and not sell:
            adx_v = float(last.get("adx15", 0) or 0)
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[TCImproved] ADX={adx_v:.0f} no signal")

        dist = float(last["dist"])
        if dist <= 0 or np.isnan(dist):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[TCImproved] invalid dist")

        p = self._p

        def _meta(side: str) -> dict:
            sl  = current_price - dist if side == "long" else current_price + dist
            tp1 = current_price + p["tp1_r"] * dist if side == "long" else current_price - p["tp1_r"] * dist
            tp2 = current_price + p["tp2_r"] * dist if side == "long" else current_price - p["tp2_r"] * dist
            return {
                "stop_loss":   sl,
                "take_profit": tp2,
                "sl_init":     sl,
                "tp1":         tp1,
                "tp2":         tp2,
                "atr":         float(last["atr"]),
                "sl_dist_pct": dist / current_price,
                "partial_pct": p["tp1_fraction"],
                "risk_pct":    0.02,
                "breakeven":   current_price,
                "rr_tp1":      p["tp1_r"],
                "rr_tp2":      p["tp2_r"],
                "one_r":       dist,
            }

        adx_v = float(last.get("adx15", 0) or 0)

        if buy:
            meta = _meta("long")
            return Signal(
                SignalType.BUY, self.symbol, current_price,
                amount=0.08, reason="buy", confidence=0.80,
                metadata=meta,
            )

        meta = _meta("short")
        return Signal(
            SignalType.SELL, self.symbol, current_price,
            amount=0.08, reason="sell", confidence=0.80,
            metadata=meta,
        )
