"""
TrendContinuation Improved v2 — 15m primary + 1h/4h MTF.

Primary: 15m candles. MTF: 1h + 4h.
Core: ADX(14, 15m) gate. Partial exit: TP1=0.5R (40%), SL→BE, TP2=2.5R (60%).

═══════════════════════════════════════════════════════════════════════════════
TWO MODES

STRICT (default, fast_mode=False) — Strict + Swing(15m) Fast Pullback
  Layer 1  4H macro trend   EMA20(4H) vs EMA50(4H)
  Layer 2  1H mid trend     EMA20(1H) vs EMA50(1H)
  Layer 3  15m swing pull   close within 0.6% of rolling 4-bar swing low/high
             Long : (close − low4) / low4  ≤ 0.6%
             Short: (high4 − close) / high4 ≤ 0.6%
             (tracks price vs recent 1-hour extreme — faster than 1H EMA20)
  Layer 4  MTF bias > ±70   vote(15m)+vote(1h)+vote(4h)
  Layer 5  ADX(14,15m) > 30
  Layer 6  micro score ≥ 4  EMA9 / EMA20 / RSI-band / Volume

FAST MODE v2 (fast_mode=True)
  Layer 1  4H macro trend   same as Strict
  Layer 2  1H mid trend     same as Strict
  Layer 3  1H EMA20 zone    |1H_close − EMA20(1H)| / EMA20(1H) ≤ 1.8%  (was 1.0%)
             wick_bounce / wick_reject: ±0.3% from EMA20 (same as before)
  Layer 4  MTF bias > ±40   (Strict = ±70, relaxed to allow more signals)
  Layer 5  ADX(14,15m) > 18 AND ADX rising (ADX[0] > ADX[1])
  Layer 6  micro score ≥ 4  same as Strict (not relaxed)
  Layer 7  Whipsaw cooldown block next 5 bars (75 min) after any signal

Exits (both modes): SL=1.2×ATR, TP1=0.5R (40% → SL→BE), TP2=2.5R (60% runner)

═══════════════════════════════════════════════════════════════════════════════
CRASH-GUARD HEALTH MONITOR (check_health(), call every ~180s)
  Fires only when BOTH:
    (a) position ≥ health_underwater_frac × 1R underwater (default 0.7R), AND
    (b) 5m momentum strongly reversed:
        price below EMA9 & EMA20 (long) + MTF bias < −flip_threshold + ADX(5m)>20
  Never fires after TP1 (runner is protected).
  bias_flip auto-adjusts to mode:
    Strict   → health_bias_flip param  (default 50.0)
    Fast v2  → bias_gate_fast         (default 40.0)

Backtest Jan–May 2026 ($50/20x/0.04%):
  BTC STRICT: 142 trades, WR 77.5%, +$212.95, MaxDD 6.4%
  XAU STRICT:  99 trades, WR 69.7%, +$67.54,  MaxDD 16.3%  (Liq×1 at 20x)
  ⚠️  Run XAU at ≤10x leverage to avoid +4.6% gap liquidations.
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
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _rma(dx.fillna(0), n)


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

    fast_mode = p.get("fast_mode", False)

    # ── Layer 1: 4H macro trend ───────────────────────────────────────────────
    ef4, es4 = _ema(out["h4_close"], p["ema_fast"]), _ema(out["h4_close"], p["ema_slow"])
    macro_up, macro_dn = ef4 > es4, ef4 < es4

    # ── Layer 2: 1H mid trend ─────────────────────────────────────────────────
    ef1, es1 = _ema(out["h1_close"], p["ema_fast"]), _ema(out["h1_close"], p["ema_slow"])
    mid_up, mid_dn = ef1 > es1, ef1 < es1

    # ── Layer 3: Pullback zone (mode-specific) ────────────────────────────────
    if fast_mode:
        # Fast Mode v2: 1H EMA20 zone, wider ±1.8%
        pullback_pct = p["pullback_pct_fast"]
        ema20_1h = ef1
        near_ema      = (out["h1_close"] - ema20_1h).abs() / ema20_1h <= pullback_pct
        wick_bounce_l = (out["h1_low"]  <= ema20_1h * 1.003) & (out["h1_close"] > ema20_1h)
        wick_reject_s = (out["h1_high"] >= ema20_1h * 0.997) & (out["h1_close"] < ema20_1h)
        at_pull_long  = near_ema | wick_bounce_l
        at_pull_short = near_ema | wick_reject_s
    else:
        # Strict: 15m swing low/high of last N bars (rolling window)
        swing_n   = p["swing_lookback"]
        swing_pct = p["swing_pct"]
        swing_low  = out["low"].rolling(swing_n).min()
        swing_high = out["high"].rolling(swing_n).max()
        # Require close >= swing_low (pullback to support, NOT breakdown below it).
        # A negative numerator (close < swing_low) would otherwise satisfy <= swing_pct.
        at_pull_long  = (out["close"] >= swing_low) & (
            (out["close"] - swing_low) / swing_low.replace(0, np.nan) <= swing_pct
        )
        at_pull_short = (out["close"] <= swing_high) & (
            (swing_high - out["close"]) / swing_high.replace(0, np.nan) <= swing_pct
        )

    # ── Layer 4: MTF composite bias (mode-specific gate) ─────────────────────
    bias_gate = p["bias_gate_fast"] if fast_mode else p["bias_gate"]
    comp_pct = _mtf_bias(out["close"], {"1h": out["h1_close"], "4h": out["h4_close"]})
    out["comp_pct"] = comp_pct
    long_bias_ok  = comp_pct > bias_gate
    short_bias_ok = comp_pct < -bias_gate

    # ── Layer 5: ADX gate (mode-specific) ────────────────────────────────────
    out["atr"]   = _atr(out, p["atr_period"])
    out["adx15"] = _adx(out, p["adx_len"])
    if fast_mode:
        # ADX > 18 AND ADX rising
        adx_rising = out["adx15"] > out["adx15"].shift(1).fillna(0)
        adx_ok = (out["adx15"] > p["adx_min_fast"]) & adx_rising
    else:
        adx_ok = out["adx15"] > p["adx_min"]

    # ── Layer 6: Micro score indicators ──────────────────────────────────────
    ema9  = _ema(out["close"], p["ema_micro"])
    ema20 = _ema(out["close"], p["ema_fast"])
    rsi15 = _rsi(out["close"], p["rsi_period"])
    volma = _sma(out["volume"], p["vol_period"])
    vol_ok = (volma > 0) & (out["volume"] >= volma * p["vol_mult"])

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

    # Diagnostic columns — each layer's pass/fail for the last bar
    out["d_macro_up"]  = macro_up.fillna(False)
    out["d_macro_dn"]  = macro_dn.fillna(False)
    out["d_mid_up"]    = mid_up.fillna(False)
    out["d_mid_dn"]    = mid_dn.fillna(False)
    out["d_pull_l"]    = at_pull_long.fillna(False)
    out["d_pull_s"]    = at_pull_short.fillna(False)
    out["d_bias_l"]    = long_bias_ok.fillna(False)
    out["d_bias_s"]    = short_bias_ok.fillna(False)
    out["d_adx_ok"]    = adx_ok.fillna(False)

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
    bias_flip is mode-aware: 50 for strict, 40 for fast mode v2 (passed by caller).
    """
    if tp1_hit:
        return ("HOLD", "post-TP1 runner — guard disabled")
    if len(df5) < max(ema_slow, 30) or len(df1h) < ema_slow or len(df4h) < ema_slow:
        return ("HOLD", "insufficient data")
    if one_r <= 0:
        return ("HOLD", "invalid one_r — guard skipped")

    px = float(df5["close"].iloc[-1])

    uw = (entry - px) / one_r if side == "long" else (px - entry) / one_r
    if uw < underwater_frac:
        return ("HOLD", f"only {uw:.2f}R underwater (< {underwater_frac}R) — safe")

    e9   = float(_ema(df5["close"], 9).iloc[-1])
    e20  = float(_ema(df5["close"], 20).iloc[-1])
    adx5 = float(_adx(df5, 14).iloc[-1])
    # Strip DatetimeIndex before combining 5m/1h/4h Series — different timestamps would
    # cause pandas index-alignment in _mtf_bias to produce mostly-NaN, disabling the guard.
    bias = float(_mtf_bias(
        pd.Series(df5["close"].values),
        {"1h": pd.Series(df1h["close"].values), "4h": pd.Series(df4h["close"].values)},
    ).iloc[-1])

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
    TrendContinuation Improved v2: 15m primary + 1h/4h MTF.
    STRICT: 15m swing pullback + ADX>30, TP2=2.5R. FAST v2: 1H EMA20 ±1.8% + bias>70 + ADX>18-rising + cooldown, TP2=3.0R.
    Partial-close: TP1=0.5R (close 40%), SL→BE, runner to TP2.
    """

    MTF_TIMEFRAMES = ["1h", "4h"]

    DEFAULTS = dict(
        ema_fast=20, ema_slow=50, ema_micro=9, rsi_period=14,
        rsi_min_buy=35.0, rsi_max_buy=75.0,
        rsi_min_sell=28.0, rsi_max_sell=65.0,
        bias_gate=70.0,
        # Strict: 15m swing pullback zone
        swing_lookback=4,    # rolling bars for swing high/low
        swing_pct=0.006,     # 0.6% max distance from swing extreme
        vol_period=20, vol_mult=1.0,
        atr_period=14, sl_mult=1.2, sl_min_pct=0.012, sl_max_pct=0.035,
        adx_len=14, adx_min=30,
        tp1_r=0.5, tp1_fraction=0.40, tp2_r=2.5,
        min_score=4.0,
        # Fast mode v2
        fast_mode=False,
        adx_min_fast=18,          # ADX threshold (was 20/30)
        adx_rising_fast=True,     # also require ADX[0] > ADX[1]
        pullback_pct_fast=0.018,  # 1H EMA20 zone ±1.8% (was 1.0%)
        bias_gate_fast=60.0,      # MTF bias gate — 60 is live-calibrated (grid winner was 70 but too strict for live market phases)
        tp2_r_fast=3.0,           # FAST runner target 3.0R (grid winner; lets trends run further)
        cooldown_bars=5,          # whipsaw cooldown: block N bars after signal
        # crash-guard
        health_guard_enabled=True,
        health_underwater_frac=0.7,
        health_bias_flip=50.0,    # used in strict mode; fast mode uses bias_gate_fast
    )

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self._p = {**self.DEFAULTS, **{k: self.params.get(k, v) for k, v in self.DEFAULTS.items()}}
        self._p["indicators"] = self.params.get("indicators", build_indicator_registry())
        self._min_primary = self.params.get("min_primary", 100)
        self._min_1h      = self.params.get("min_1h",       60)
        self._min_4h      = self.params.get("min_4h",       55)
        self._cooldown_remaining = 0  # bars remaining in whipsaw cooldown (fast mode)

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
        bias_flip: uses bias_gate_fast (40) in fast mode, health_bias_flip (50) in strict.
        """
        if not self._p["health_guard_enabled"]:
            return ("HOLD", "health guard disabled")
        if not candles_5m or not candles_1h or not candles_4h:
            return ("HOLD", "no candle data")
        df5  = self._to_df(candles_5m)
        df1h = self._to_df(candles_1h)
        df4h = self._to_df(candles_4h)
        # Fast mode entered at bias > 40 → reversed when bias < -40
        bias_flip = (self._p["bias_gate_fast"] if self._p.get("fast_mode")
                     else self._p["health_bias_flip"])
        return check_health(
            side, entry, one_r, tp1_hit, df5, df1h, df4h,
            underwater_frac=self._p["health_underwater_frac"],
            bias_flip=bias_flip,
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

        # ── Whipsaw cooldown (fast mode only) ─────────────────────────────────
        if self._p.get("fast_mode") and self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[TCImproved] whipsaw cooldown: {self._cooldown_remaining} bars left")

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
            adx_v  = float(last.get("adx15", 0) or 0)
            bias_v = float(last.get("comp_pct", 0) or 0)
            sc_l   = float(last.get("score_long", 0) or 0)
            sc_s   = float(last.get("score_short", 0) or 0)
            gate   = p["bias_gate_fast"] if p.get("fast_mode") else p["bias_gate"]
            _y = lambda k: "✓" if last.get(k) else "✗"
            long_layers  = f"4H{_y('d_macro_up')} 1H{_y('d_mid_up')} pull{_y('d_pull_l')} bias{_y('d_bias_l')} adx{_y('d_adx_ok')}"
            short_layers = f"4H{_y('d_macro_dn')} 1H{_y('d_mid_dn')} pull{_y('d_pull_s')} bias{_y('d_bias_s')} adx{_y('d_adx_ok')}"
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[TCImproved] bias={bias_v:.0f}(gate±{gate:.0f}) ADX={adx_v:.0f} "
                          f"score={sc_l:.0f}L/{sc_s:.0f}S | L:{long_layers} | S:{short_layers}")

        dist = float(last["dist"])
        if dist <= 0 or np.isnan(dist):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[TCImproved] invalid dist")

        # Arm whipsaw cooldown for fast mode
        if self._p.get("fast_mode") and self._p.get("cooldown_bars", 0) > 0:
            self._cooldown_remaining = self._p["cooldown_bars"]

        p = self._p

        # FAST v2 runs the runner to 3.0R; STRICT keeps 2.5R.
        tp2_r = p.get("tp2_r_fast", p["tp2_r"]) if p.get("fast_mode") else p["tp2_r"]

        def _meta(side: str) -> dict:
            sl  = current_price - dist if side == "long" else current_price + dist
            tp1 = current_price + p["tp1_r"] * dist if side == "long" else current_price - p["tp1_r"] * dist
            tp2 = current_price + tp2_r * dist if side == "long" else current_price - tp2_r * dist
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
                "rr_tp2":      tp2_r,
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
