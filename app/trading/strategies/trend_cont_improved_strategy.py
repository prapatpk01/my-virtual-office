"""
TrendContinuation Improved v2 — 15m primary + 1h/4h MTF.

Primary: 15m candles. MTF: 1h + 4h.
Core: ADX(14, 15m) > 30 gate. Partial exit: TP1=0.5R (40%), SL→BE, TP2=2.5R (60%).

═══════════════════════════════════════════════════════════════════════════════
CHANGES IN v2 vs v1:

1. MODULAR INDICATOR SYSTEM (INDICATORS dict + WEIGHTS):
   Every entry sub-signal is a named, toggleable component with a weight.
   Set enabled=False to drop one, change weight to reweight. The entry gate
   requires the SUM of enabled weights that pass >= min_score.

2. OPTIONAL "FAST MODE" (fast_mode param, default False):
   Loosens ADX 30→20 and pullback 1.0%→1.5% for more signals.
   WARNING: BTC WR 77.5%→71.4%, XAU goes NEGATIVE. Default stays STRICT.

3. CRASH-GUARD HEALTH MONITOR (check_health(), call every ~180s):
   Only fires when position is BOTH:
     (a) underwater past health_underwater_frac × 1R (default 0.7R), AND
     (b) 5m momentum strongly against (price beyond both EMAs + MTF bias
         flipped past ±health_bias_flip + ADX(5m)>20).
   Never fires on a runner that already banked TP1.
   Backtest effect (Jan–May 2026, $50/20x/0.04%):
     XAU: PnL +$67.98 → +$155.43, MaxDD −15.5% → −10.0%  ← clear win
     BTC: PnL +$212.94 → +$182.58                         ← slight drag
   Default health_guard_enabled=True (protects XAU gaps/sharp reversals).
   Consider False for BTC-only setups.
   ⚠️ XAU liquidated once at 20x on a +4.6% 5m candle — run XAU at ≤10x.
═══════════════════════════════════════════════════════════════════════════════
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


# ── MODULAR MICRO-INDICATOR REGISTRY ──────────────────────────────────────────

def build_indicator_registry():
    return {
        "above_ema9": dict(
            enabled=True, weight=1.0,
            long =lambda c: c["close"] > c["ema9"],
            short=lambda c: c["close"] < c["ema9"],
            desc="15m close vs EMA9 (micro momentum)",
        ),
        "above_ema20": dict(
            enabled=True, weight=1.0,
            long =lambda c: c["close"] > c["ema20"],
            short=lambda c: c["close"] < c["ema20"],
            desc="15m close vs EMA20 (local trend)",
        ),
        "rsi_band": dict(
            enabled=True, weight=1.0,
            long =lambda c: (c["rsi15"] >= c["rsi_min_buy"]) & (c["rsi15"] <= c["rsi_max_buy"]),
            short=lambda c: (c["rsi15"] >= c["rsi_min_sell"]) & (c["rsi15"] <= c["rsi_max_sell"]),
            desc="15m RSI inside allowed band",
        ),
        "volume_ok": dict(
            enabled=True, weight=1.0,
            long =lambda c: c["vol_ok"],
            short=lambda c: c["vol_ok"],
            desc="15m volume >= MA20 * vol_mult",
        ),
        # ─ EXTRA SIGNALS (disabled by default) ──────────────────────────────
        "macd_align": dict(
            enabled=False, weight=1.0,
            long =lambda c: c["macd"] > c["macd_signal"],
            short=lambda c: c["macd"] < c["macd_signal"],
            desc="15m MACD line vs signal (extra momentum confirm)",
        ),
        "ema9_slope": dict(
            enabled=False, weight=0.5,
            long =lambda c: c["ema9"] > c["ema9"].shift(2),
            short=lambda c: c["ema9"] < c["ema9"].shift(2),
            desc="15m EMA9 rising/falling over 2 bars",
        ),
    }


# ── Core computation ─────────────────────────────────────────────────────────

def _compute(df15: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame, p: dict) -> pd.DataFrame:
    out = df15.copy()
    out = out.join(_merge_htf(df15, df1h, "1h", "h1"))
    out = out.join(_merge_htf(df15, df4h, "4h", "h4"))

    adx_min      = p["adx_min_fast"]      if p.get("fast_mode") else p["adx_min"]
    pullback_pct = p["pullback_pct_fast"] if p.get("fast_mode") else p["pullback_pct"]

    ef4, es4 = _ema(out["h4_close"], p["ema_fast"]), _ema(out["h4_close"], p["ema_slow"])
    macro_up, macro_dn = ef4 > es4, ef4 < es4

    ef1, es1 = _ema(out["h1_close"], p["ema_fast"]), _ema(out["h1_close"], p["ema_slow"])
    mid_up, mid_dn = ef1 > es1, ef1 < es1

    ema20_1h = ef1
    near_ema       = (out["h1_close"] - ema20_1h).abs() / ema20_1h <= pullback_pct
    wick_bounce_l  = (out["h1_low"]  <= ema20_1h * 1.003) & (out["h1_close"] > ema20_1h)
    wick_reject_s  = (out["h1_high"] >= ema20_1h * 0.997) & (out["h1_close"] < ema20_1h)
    at_pull_long   = near_ema | wick_bounce_l
    at_pull_short  = near_ema | wick_reject_s

    comp_pct = _mtf_bias(out["close"], {"1h": out["h1_close"], "4h": out["h4_close"]})
    out["comp_pct"] = comp_pct
    long_bias_ok   = comp_pct > p["bias_gate"]
    short_bias_ok  = comp_pct < -p["bias_gate"]

    ema9  = _ema(out["close"], p["ema_micro"])
    ema20 = _ema(out["close"], p["ema_fast"])
    rsi15 = _rsi(out["close"], p["rsi_period"])
    volma = _sma(out["volume"], p["vol_period"])
    out["atr"]   = _atr(out, p["atr_period"])
    out["adx15"] = _adx(out, p["adx_len"])
    vol_ok = (volma > 0) & (out["volume"] >= volma * p["vol_mult"])
    adx_ok = out["adx15"] > adx_min

    macd_line = _ema(out["close"], 12) - _ema(out["close"], 26)
    macd_sig  = _ema(macd_line, 9)

    ctx = dict(
        close=out["close"], ema9=ema9, ema20=ema20, rsi15=rsi15, vol_ok=vol_ok,
        macd=macd_line, macd_signal=macd_sig,
        rsi_min_buy=p["rsi_min_buy"], rsi_max_buy=p["rsi_max_buy"],
        rsi_min_sell=p["rsi_min_sell"], rsi_max_sell=p["rsi_max_sell"],
    )

    registry = p["indicators"]
    score_long  = pd.Series(0.0, index=out.index)
    score_short = pd.Series(0.0, index=out.index)
    for name, ind in registry.items():
        if not ind.get("enabled", True):
            continue
        w = ind.get("weight", 1.0)
        score_long  = score_long  + ind["long"](ctx).fillna(False).astype(float)  * w
        score_short = score_short + ind["short"](ctx).fillna(False).astype(float) * w
    out["score_long"]  = score_long
    out["score_short"] = score_short

    out["final_buy"]  = (macro_up & mid_up  & at_pull_long  & long_bias_ok  &
                          (score_long  >= p["min_score"]) & adx_ok).fillna(False)
    out["final_sell"] = (macro_dn  & mid_dn  & at_pull_short & short_bias_ok &
                          (score_short >= p["min_score"]) & adx_ok).fillna(False)

    dist = (out["atr"] * p["sl_mult"]).clip(
        lower=out["close"] * p["sl_min_pct"],
        upper=out["close"] * p["sl_max_pct"],
    )
    out["dist"] = dist
    return out


# ── CRASH-GUARD HEALTH MONITOR ────────────────────────────────────────────────

def check_health(side: str, entry: float, one_r: float, tp1_hit: bool,
                 df5: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame,
                 underwater_frac: float = 0.7, bias_flip: float = 50.0,
                 ema_fast: int = 20, ema_slow: int = 50) -> tuple[str, str]:
    """
    Returns ('CLOSE', reason) only when position is BOTH deeply underwater AND
    momentum has strongly reversed. Never acts on a runner after TP1.
    """
    if tp1_hit:
        return ("HOLD", "post-TP1 runner — guard disabled")
    if len(df5) < max(ema_slow, 30) or len(df1h) < ema_slow or len(df4h) < ema_slow:
        return ("HOLD", "insufficient data")

    px = float(df5["close"].iloc[-1])

    uw = (entry - px) / one_r if side == "long" else (px - entry) / one_r
    if uw < underwater_frac:
        return ("HOLD", f"only {uw:.2f}R underwater (< {underwater_frac}R) — safe")

    e9   = float(_ema(df5["close"], 9).iloc[-1])
    e20  = float(_ema(df5["close"], 20).iloc[-1])
    adx5 = float(_adx(df5, 14).iloc[-1])
    bias = float(_mtf_bias(df5["close"], {"1h": df1h["close"], "4h": df4h["close"]}).iloc[-1])

    if side == "long":
        against = (px < e9) and (px < e20) and (bias < -bias_flip) and (adx5 > 20)
    else:
        against = (px > e9) and (px > e20) and (bias > bias_flip) and (adx5 > 20)

    if against:
        return ("CLOSE",
                f"CRASH-GUARD: {uw:.2f}R underwater + momentum reversed (bias={bias:.0f}, adx={adx5:.0f})")
    return ("HOLD",
            f"{uw:.2f}R underwater but momentum not confirmed against — hold for SL/recovery")


# ── BaseStrategy wrapper ─────────────────────────────────────────────────────

class TrendContImprovedStrategy(BaseStrategy):
    """
    TrendContinuation Improved v2: 15m primary + 1h/4h MTF, ADX(15m)>30 filter.
    Partial-close: TP1=0.5R (close 40%), SL→BE, TP2=2.5R (60% runner).
    Modular micro-indicators + optional fast mode + crash-guard health monitor.
    """

    MTF_TIMEFRAMES = ["1h", "4h"]

    DEFAULTS = dict(
        ema_fast=20, ema_slow=50, ema_micro=9, rsi_period=14,
        rsi_min_buy=35.0, rsi_max_buy=75.0,
        rsi_min_sell=28.0, rsi_max_sell=65.0,
        bias_gate=70.0,
        pullback_pct=0.010,
        vol_period=20, vol_mult=1.0,
        atr_period=14, sl_mult=1.2, sl_min_pct=0.012, sl_max_pct=0.035,
        adx_len=14, adx_min=30,
        tp1_r=0.5, tp1_fraction=0.40, tp2_r=2.5,
        min_score=4.0,
        # fast mode
        fast_mode=False,
        adx_min_fast=20, pullback_pct_fast=0.015,
        # crash-guard
        health_guard_enabled=True,
        health_underwater_frac=0.7,
        health_bias_flip=50.0,
    )

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self._p = {**self.DEFAULTS, **{k: self.params.get(k, v) for k, v in self.DEFAULTS.items()}}
        self._p["indicators"] = self.params.get("indicators", build_indicator_registry())
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

    def monitor_position(self, side: str, entry: float, one_r: float, tp1_hit: bool,
                         candles_5m: list, candles_1h: list, candles_4h: list) -> tuple[str, str]:
        """
        Crash-guard health check. Returns ('CLOSE'|'HOLD', reason).
        Call from bot's monitor loop every ~180s.
        """
        if not self._p["health_guard_enabled"]:
            return ("HOLD", "health guard disabled")
        if not candles_5m or not candles_1h or not candles_4h:
            return ("HOLD", "no candle data")
        df5  = self._to_df(candles_5m)
        df1h = self._to_df(candles_1h)
        df4h = self._to_df(candles_4h)
        return check_health(
            side, entry, one_r, tp1_hit, df5, df1h, df4h,
            underwater_frac=self._p["health_underwater_frac"],
            bias_flip=self._p["health_bias_flip"],
            ema_fast=self._p["ema_fast"], ema_slow=self._p["ema_slow"],
        )

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

        last = computed.iloc[-2]  # [-1] is still-forming bar

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

        if buy:
            return Signal(
                SignalType.BUY, self.symbol, current_price,
                amount=0.08, reason="buy", confidence=0.80,
                metadata=_meta("long"),
            )

        return Signal(
            SignalType.SELL, self.symbol, current_price,
            amount=0.08, reason="sell", confidence=0.80,
            metadata=_meta("short"),
        )
