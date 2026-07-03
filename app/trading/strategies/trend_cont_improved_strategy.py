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
  Layer 3  1H EMA20 zone    |1H_close − EMA20(1H)| / EMA20(1H) ≤ 2.5%  (TCI_PULLBACK_PCT_FAST)
             wick_bounce / wick_reject: low/high within 0.3% of EMA20
  Layer 4  MTF bias > ±60   live-calibrated (grid winner was 70 but caused signal drought)
  Layer 5  ADX(14,15m) > 18 AND ADX rising (ADX[0] > ADX[1])
  Layer 6  micro score ≥ 4  same as Strict (not relaxed)
  Layer 7  Whipsaw cooldown block next 5 bars (75 min) after any signal

Exits: FAST TP2=3.0R, STRICT TP2=2.5R. Both: SL=1.2×ATR, TP1=0.5R (40%→SL→BE).

═══════════════════════════════════════════════════════════════════════════════
CRASH-GUARD HEALTH MONITOR (check_health(), call every ~180s)
  Fires only when BOTH:
    (a) position ≥ health_underwater_frac × 1R underwater (default 0.7R), AND
    (b) 5m momentum strongly reversed:
        price below EMA9 & EMA20 (long) + MTF bias < −flip_threshold + ADX(5m)>20
  Never fires after TP1 (runner is protected).
  bias_flip auto-adjusts to mode:
    Strict   → health_bias_flip param  (default 50.0)
    Fast v2  → bias_gate_fast         (default 60.0)

Backtest Jan–May 2026 ($50/20x/0.04%):
  BTC STRICT: 142 trades, WR 77.5%, +$212.95, MaxDD 6.4%
  XAU STRICT:  99 trades, WR 69.7%, +$67.54,  MaxDD 16.3%  (Liq×1 at 20x)
  ⚠️  Run XAU at ≤10x leverage to avoid +4.6% gap liquidations.
═══════════════════════════════════════════════════════════════════════════════
"""
import time
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
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    # loss==0 & gain>0 → pure up-streak → RSI 100; flat (gain==0 & loss==0) → neutral 50
    rsi = rsi.where(~((loss == 0) & (gain > 0)), 100.0)
    return rsi.fillna(50.0)


def _true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)


def _wma(s: pd.Series, n: int) -> pd.Series:
    weights = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def _hma(s: pd.Series, n: int) -> pd.Series:
    """Hull Moving Average — faster and smoother than EMA/SMA of same period."""
    sqrt_n = max(2, int(round(n ** 0.5)))
    return _wma(2 * _wma(s, n // 2) - _wma(s, n), sqrt_n)


def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _rma(_true_range(df), period)


def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """
    Supertrend — ATR-band volatility indicator. Returns +1 (bullish) / -1 (bearish).
    Completely different from MA family: flips only when price breaches the band,
    so small wicks don't cause noise. Complements HMA/EMA components.
    """
    atr   = _atr(df, period).to_numpy()
    hl2   = ((df["high"] + df["low"]) / 2).to_numpy()
    close = df["close"].to_numpy()
    n     = len(close)

    ub = hl2 + multiplier * atr   # basic upper band
    lb = hl2 - multiplier * atr   # basic lower band
    fu = ub.copy()                # final upper band
    fl = lb.copy()                # final lower band
    st = np.ones(n)               # supertrend direction: +1 bull / -1 bear

    for i in range(1, n):
        fu[i] = ub[i] if (ub[i] < fu[i-1] or close[i-1] > fu[i-1]) else fu[i-1]
        fl[i] = lb[i] if (lb[i] > fl[i-1] or close[i-1] < fl[i-1]) else fl[i-1]
        if   close[i] > fu[i-1]: st[i] =  1
        elif close[i] < fl[i-1]: st[i] = -1
        else:                     st[i] =  st[i-1]

    return pd.Series(st, index=df.index)


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


def _roc(s: pd.Series, n: int = 9) -> pd.Series:
    return (s - s.shift(n)) / s.shift(n).replace(0, np.nan) * 100


def _zlema(s: pd.Series, n: int) -> pd.Series:
    """Zero-Lag EMA: 2×EMA(n) − EMA(EMA(n)) — roughly halves EMA lag."""
    ema1 = _ema(s, n)
    return 2 * ema1 - _ema(ema1, n)


def _chop_index(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Choppiness Index (0–100). LOW <50 = trending; HIGH >61 = choppy/ranging."""
    tr_sum = _true_range(df).rolling(n).sum()
    hl_n   = df["high"].rolling(n).max() - df["low"].rolling(n).min()
    ci     = 100 * np.log10(tr_sum / hl_n.replace(0, np.nan)) / np.log10(n)
    return ci.fillna(50.0)


def _atr_percentile(atr_series: pd.Series, window: int = 100) -> pd.Series:
    """Rolling percentile rank of ATR14 (0–100). <20 = quiet market, >80 = volatile."""
    return atr_series.rolling(window).rank(pct=True).fillna(0.5) * 100


def _market_regime(
    adx: pd.Series, chop_1h: pd.Series, atr_pct: pd.Series,
    ema20: pd.Series, rsi15: pd.Series, close: pd.Series, p: dict,
) -> tuple:
    """
    Classify each 15m bar into a market regime.

    Regimes (priority HIGH → LOW, last written wins):
      TREND        ADX > 18 AND Chop(1H) < 50      — normal trending
      STRONG_TREND ADX > 30 AND Chop(1H) < 38      — best conditions
      HIGH_VOL     ATR > 80th pct                  — volatile, allow but widen SL
      EXHAUSTION   ADX fading + RSI extreme + chop rising — trend dying, skip
      LOW_VOL      ATR < 20th pct                  — dead market, skip
      RANGE        (default)                        — skip

    Returns (regime_series, entry_ok_bool, sl_mult_float_series).
    """
    strong_adx  = float(p.get("regime_strong_adx",   30))
    trend_adx   = float(p.get("regime_trend_adx",    18))
    low_pct     = float(p.get("regime_low_vol_pct",  20))
    high_pct    = float(p.get("regime_high_vol_pct", 80))
    strong_chop = float(p.get("regime_strong_chop",  38))
    trend_chop  = float(p.get("chop_threshold",      50.0))
    sl_hv       = float(p.get("regime_sl_high_vol",  1.5))

    adx_peak   = adx.rolling(10).max().shift(1)
    adx_fading = adx < adx_peak * 0.85            # ADX dropped >15% from recent 10-bar peak
    rsi_ext    = (rsi15 > 75) | (rsi15 < 25)
    ema20_flat = (ema20 - ema20.shift(3)).abs() / close.replace(0, np.nan) < 0.0008
    chop_up    = chop_1h > chop_1h.shift(3)       # chop worsening (trending → ranging)

    regime = pd.Series("RANGE", index=close.index, dtype=object)
    regime[(adx > trend_adx)  & (chop_1h < trend_chop)]   = "TREND"
    regime[(adx > strong_adx) & (chop_1h < strong_chop)]  = "STRONG_TREND"
    regime[atr_pct > high_pct]                             = "HIGH_VOL"
    regime[adx_fading & rsi_ext & (ema20_flat | chop_up)]  = "EXHAUSTION"
    regime[atr_pct < low_pct]                              = "LOW_VOL"

    entry_ok = regime.isin(["TREND", "STRONG_TREND", "HIGH_VOL"])
    sl_mult  = pd.Series(1.0, index=close.index)
    sl_mult[regime == "HIGH_VOL"] = sl_hv

    return regime, entry_ok, sl_mult


def _htf_dir(ef: pd.Series, es: pd.Series, close: pd.Series,
             mode: str = "cross", slope_bars: int = 2, sep_guard: bool = False,
             slope_pct: float = 0.0):
    """HTF trend direction (up, dn) for the macro/mid gates.

    'cross'  : EMA_fast vs EMA_slow (legacy binary; lags a turn most).
    'slope'  : EMA_fast direction over slope_bars — flips ~1 fast-EMA length
               sooner, ignores the slow EMA entirely (more whipsaw).
    'early'  : full cross OR (fast-EMA turning AND price on that side of it) —
               lets the gate flip before the 20/50 cross completes, with a
               price-confirm guard to cut the worst whipsaw.

    slope_pct: when > 0 in 'early'/'slope' mode, the early flip requires
               EMA_fast to move at least slope_pct% in one bar (quantified
               momentum). 0.15 recommended — filters micro-oscillation turns.

    sep_guard: when True, the *early/slope* trigger only fires while the
    EMA_fast/EMA_slow gap is WIDENING (trend accelerating, not coiled) —
    plugs the whipsaw hole that earlier confirmation opens. The full cross
    branch is never gated (already confirmed).
    """
    if sep_guard:
        sep = (ef - es).abs() / close.replace(0, np.nan)
        sep_ok = sep > sep.shift(slope_bars)         # gap expanding
    else:
        sep_ok = True
    if mode == "slope":
        if slope_pct > 0:
            sp = (ef - ef.shift(1)) / ef.shift(1).replace(0, np.nan) * 100
            up = (sp > slope_pct) & sep_ok
            dn = (sp < -slope_pct) & sep_ok
        else:
            up = (ef > ef.shift(slope_bars)) & sep_ok
            dn = (ef < ef.shift(slope_bars)) & sep_ok
    elif mode == "early":
        cross_up, cross_dn = ef > es, ef < es
        if slope_pct > 0:
            sp = (ef - ef.shift(1)) / ef.shift(1).replace(0, np.nan) * 100
            slope_up = (sp > slope_pct) & (close > ef) & sep_ok
            slope_dn = (sp < -slope_pct) & (close < ef) & sep_ok
        else:
            slope_up = (ef > ef.shift(slope_bars)) & (close > ef) & sep_ok
            slope_dn = (ef < ef.shift(slope_bars)) & (close < ef) & sep_ok
        up = cross_up | slope_up
        dn = cross_dn | slope_dn
    else:  # 'cross' (default — legacy behaviour)
        up, dn = ef > es, ef < es
    return up, dn


def _mtf_bias(primary_close: pd.Series, aligned: dict,
              ema_fast: int = 20, ema_slow: int = 50,
              rsi_period: int = 14, rsi_bull: float = 55.0, rsi_bear: float = 45.0,
              include_primary: bool = True) -> pd.Series:
    """Composite MTF bias −100…+100.

    include_primary=False drops the primary (15m) vote so the bias reflects only
    the higher timeframes (1h+4h) — a cleaner directional gate without 15m noise
    (the 15m TF is already filtered separately by the ADX/score/MACD gates).
    """
    def _vote(s: pd.Series) -> pd.Series:
        ef, es, r = _ema(s, ema_fast), _ema(s, ema_slow), _rsi(s, rsi_period)
        v = (np.where(s > ef, 1, -1) + np.where(ef > es, 1, -1) +
             np.where(r > rsi_bull, 1, np.where(r < rsi_bear, -1, 0)))
        return pd.Series(v, index=s.index)
    votes = [_vote(c) for c in aligned.values()]
    if include_primary:
        votes = [_vote(primary_close)] + votes
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
#
# CLASSIC mode (4 components, min 4/4):  above_ema9, above_ema20, rsi_band, volume_ok
# SJ HYBRID mode (5 components, min 4/5): hma20_bull, ema5_sma9, rsi_band, volume_ok,
#   breakout_hh10 — replaces EMA9/EMA20 with faster HMA20/EMA5>SMA9 + breakout
#
# Backtest Jan-May 2026 ($50×20x): Classic=+$90 combined; SJ Hybrid=+$103 combined
# SJ Hybrid improves XAU +134% ($38→$89) while keeping BTC profitable (+$14 vs +$52).

def build_indicator_registry(sj_scoring: bool = False, extended: bool = False,
                             roc9: bool = False, vol_expansion: bool = False,
                             bos: bool = False, zlema: bool = False,
                             price_action: bool = False, rsi_div: bool = False,
                             rel_vol: bool = False, obv: bool = False,
                             ema_slope: bool = False, adx_rising_score: bool = False,
                             atr_expansion: bool = False, ema5_sma9: bool = True,
                             macd_hist_score: bool = False):
    if sj_scoring:
        # Weighted Confidence Score (0-100 scale), min_score=60.
        # Base (3 components, max 35 pts): hma_bull(15) rsi_band(10) breakout_hh10(10)
        # Optional default-on (8 more, max 65 pts): roc9(10) obv(5) bos(15) rel_vol(10)
        #   ema_slope(10) adx_rising(5) atr_expansion(5) macd_hist(5)
        # ema5_sma9(5) is off by default — overlaps with hma_bull (both measure
        # fast-vs-slow direction) and adds no distinct information.
        # macd_hist(5) gives the confidence score explicit credit for MACD histogram
        # sign — the same signal simple_fast_entry already gates on for timing.
        # Total max = 100 | ≥90 strong, ≥70 normal, ≥60 small, <60 skip
        reg = {
            "hma_bull": dict(
                enabled=True, weight=15.0,
                long =lambda c: c["close"] > c["hma"],
                short=lambda c: c["close"] < c["hma"],
                desc="HMA16 trend direction — primary trend gate (15 pts)",
            ),
            "rsi_band": dict(
                enabled=True, weight=10.0,
                long =lambda c: (c["rsi15"] >= c["rsi_min_buy"]) & (c["rsi15"] <= c["rsi_max_buy"]),
                short=lambda c: (c["rsi15"] >= c["rsi_min_sell"]) & (c["rsi15"] <= c["rsi_max_sell"]),
                desc="RSI inside non-overbought/oversold band (10 pts)",
            ),
            "breakout_hh10": dict(
                enabled=True, weight=10.0,
                long =lambda c: c["close"] > c["hh10"],
                short=lambda c: c["close"] < c["ll10"],
                desc="Close breaks N-bar high/low — fast entry timing (10 pts)",
            ),
        }
        if ema5_sma9:
            reg["ema5_sma9"] = dict(
                enabled=True, weight=5.0,
                long =lambda c: c["ema5"] > c["sma9"],
                short=lambda c: c["ema5"] < c["sma9"],
                desc="EMA5 vs SMA9 momentum bonus — fast crossover (5 pts)",
            )
        if roc9:
            reg["roc9"] = dict(
                enabled=True, weight=10.0,
                long =lambda c: c["roc9"] > 0,
                short=lambda c: c["roc9"] < 0,
                desc="ROC(9) price momentum direction (10 pts)",
            )
        if vol_expansion:
            reg["vol_expansion"] = dict(
                enabled=True, weight=5.0,
                long =lambda c: c["vol_expansion_ok"],
                short=lambda c: c["vol_expansion_ok"],
                desc="Volume > MA20 × 1.15 (5 pts)",
            )
        if obv:
            reg["obv_trend"] = dict(
                enabled=True, weight=5.0,
                long =lambda c: c["obv"] > c["obv_ema20"],
                short=lambda c: c["obv"] < c["obv_ema20"],
                desc="OBV above EMA20 — smart money flow direction (5 pts)",
            )
        if ema_slope:
            reg["ema_slope"] = dict(
                enabled=True, weight=10.0,
                long =lambda c: c["ema_slope_ok"],
                short=lambda c: ~c["ema_slope_ok"],
                desc="EMA20 & EMA50 both rising — structural trend quality (10 pts)",
            )
        if adx_rising_score:
            reg["adx_rising"] = dict(
                enabled=True, weight=5.0,
                long =lambda c: c["adx_rising_ok"],
                short=lambda c: c["adx_rising_ok"],
                desc="ADX rising over 2 bars — trend is strengthening (5 pts)",
            )
        if atr_expansion:
            reg["atr_expansion"] = dict(
                enabled=True, weight=5.0,
                long =lambda c: c["atr_expand_ok"],
                short=lambda c: c["atr_expand_ok"],
                desc="ATR14 above ATR_MA20 — volatility expanding, not consolidating (5 pts)",
            )
        if macd_hist_score:
            reg["macd_hist"] = dict(
                enabled=True, weight=5.0,
                long =lambda c: c["macd"] > c["macd_signal"],
                short=lambda c: c["macd"] < c["macd_signal"],
                desc="MACD line vs signal (histogram sign) — momentum confirmation (5 pts)",
            )
        if extended:
            # SJ Extended: market structure + FVG (suggest min_score=5-6 out of 9+)
            reg.update({
                "market_struct": dict(
                    enabled=True, weight=1.0,
                    long =lambda c: c["sma20_rising"] & (c["close"] > c["sma20"]),
                    short=lambda c: (~c["sma20_rising"]) & (c["close"] < c["sma20"]),
                    desc="SMA20 rising + close above it (structural bullish/bearish context)",
                ),
                "fvg_confirm": dict(
                    enabled=True, weight=1.0,
                    long =lambda c: c["fvg_bull"],
                    short=lambda c: c["fvg_bear"],
                    desc="Bullish/bearish FVG (3-candle imbalance) formed in last 5 bars",
                ),
            })
        if bos:
            reg["bos"] = dict(
                enabled=True, weight=15.0,
                long =lambda c: c["bos_bull"],
                short=lambda c: c["bos_bear"],
                desc="Break of Structure: close exceeds N-bar swing high/low (15 pts)",
            )
        if zlema:
            reg["zlema_dir"] = dict(
                enabled=True, weight=5.0,
                long =lambda c: c["zlema9"] > c["zlema20"],
                short=lambda c: c["zlema9"] < c["zlema20"],
                desc="ZLEMA9 vs ZLEMA20 — zero-lag trend alignment (5 pts)",
            )
        if price_action:
            reg["price_action"] = dict(
                enabled=True, weight=5.0,
                long =lambda c: c["pa_bull"],
                short=lambda c: c["pa_bear"],
                desc="Pin bar / engulfing candle confirmation (5 pts)",
            )
        if rsi_div:
            reg["rsi_div"] = dict(
                enabled=True, weight=5.0,
                long =lambda c: c["hidden_bull_div"],
                short=lambda c: c["hidden_bear_div"],
                desc="Hidden RSI divergence — continuation signal (5 pts)",
            )
        if rel_vol:
            reg["rel_vol"] = dict(
                enabled=True, weight=10.0,
                long =lambda c: c["rel_vol"] >= c["rel_vol_min"],
                short=lambda c: c["rel_vol"] >= c["rel_vol_min"],
                desc="Relative volume ≥ rel_vol_min — the single volume quality gate (10 pts)",
            )
        return reg
    # Classic 4-component registry (default)
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
    htf_mode  = p.get("htf_macro_mode", "cross")
    slope_bars = int(p.get("htf_slope_bars", 2))
    slope_pct  = float(p.get("htf_slope_pct", 0.15))
    sep_guard  = bool(p.get("htf_sep_guard", False))
    ef4, es4 = _ema(out["h4_close"], p["ema_fast"]), _ema(out["h4_close"], p["ema_slow"])
    macro_up, macro_dn = _htf_dir(ef4, es4, out["h4_close"], htf_mode, slope_bars, sep_guard, slope_pct)

    # ── 4H Regime Mode routing ────────────────────────────────────────────────
    # htf_auto_regime=True: automatically selects regime by mode:
    #   fast_mode=True  → 2/3 vote (slope + ADX + price side) — fast response
    #   fast_mode=False → weighted ≥65 (slope40+ADX25+ATR15+ER20) — strict/quality
    # Explicit flags (htf_regime_4h, htf_regime_4h_mode) still override when set.
    _auto = p.get("htf_auto_regime", False)
    _use_23score  = p.get("htf_regime_4h", False) or (_auto and fast_mode)
    _use_weighted = (p.get("htf_regime_4h_mode") == "weighted") or (_auto and not fast_mode)

    # 2/3 vote: fires when ≥2 of [slope | ADX>18 | price side of EMA20]
    if _use_23score:
        _adx_len4  = int(p.get("htf_regime_adx_len", 14))
        _adx_min4  = float(p.get("htf_regime_adx_min", 18))
        adx4h_r    = _adx(
            out[["h4_high", "h4_low", "h4_close"]].rename(
                columns={"h4_high": "high", "h4_low": "low", "h4_close": "close"}),
            _adx_len4)
        adx4h_ok_r = adx4h_r > _adx_min4

        ef4_slope_up = ef4 > ef4.shift(2)   # EMA20 higher than 2 bars ago
        ef4_slope_dn = ef4 < ef4.shift(2)   # EMA20 lower than 2 bars ago
        px_above_ema = out["h4_close"] > ef4
        px_below_ema = out["h4_close"] < ef4

        score_up_r = (ef4_slope_up.astype(float) + adx4h_ok_r.astype(float)
                      + px_above_ema.astype(float))
        score_dn_r = (ef4_slope_dn.astype(float) + adx4h_ok_r.astype(float)
                      + px_below_ema.astype(float))

        macro_up = (score_up_r >= 2).fillna(False)
        macro_dn = (score_dn_r >= 2).fillna(False)

    # Weighted score: slope40 + ADX25 + ATR15 + ER20 → ≥threshold (default 65)
    # Computed on ACTUAL 4H bars then forward-filled to 15m to avoid artifacts.
    if _use_weighted:
        _adx_len4w = int(p.get("htf_regime_adx_len", 14))

        # [1] EMA20 slope over 3 actual 4H bars
        ef4_actual = _ema(df4h["close"], 20)
        slope3_pct_4h = (ef4_actual - ef4_actual.shift(3)) / ef4_actual.shift(3).abs() * 100
        slope_score_4h = np.where(slope3_pct_4h.abs() > 0.15, 40.0, 0.0)

        # [2] ADX strength buckets
        adx4h_w = _adx(df4h, _adx_len4w)
        adx_score_4h = np.where(adx4h_w < 18, 0.0,
                       np.where(adx4h_w < 25, 15.0,
                       np.where(adx4h_w <= 45, 25.0, 10.0)))

        # [3] ATR14/ATR50 expansion (actual 4H bars only)
        atr14_4h = _atr(df4h, 14)
        atr50_4h = _atr(df4h, 50)
        atr_ratio_4h = atr14_4h / atr50_4h.replace(0, np.nan)
        atr_score_4h = np.where(atr_ratio_4h > 1.10, 15.0,
                       np.where(atr_ratio_4h >= 0.90, 5.0, 0.0))

        # [4] Efficiency Ratio: directional move / total path (20 4H bars)
        close_4h = df4h["close"]
        er_dir_4h = (close_4h - close_4h.shift(20)).abs()
        er_vol_4h = close_4h.diff().abs().rolling(20).sum()
        er_4h = er_dir_4h / er_vol_4h.replace(0, np.nan)
        er_score_4h = np.where(er_4h > 0.35, 20.0,
                      np.where(er_4h >= 0.20, 10.0, 0.0))

        trend_score_4h = pd.Series(
            slope_score_4h + adx_score_4h + atr_score_4h + er_score_4h,
            index=df4h.index).fillna(0)

        _w_thresh = float(p.get("htf_regime_4h_threshold", 70))
        bull_4h = pd.Series((slope3_pct_4h > 0.15) & (trend_score_4h >= _w_thresh), index=df4h.index)
        bear_4h = pd.Series((slope3_pct_4h < -0.15) & (trend_score_4h >= _w_thresh), index=df4h.index)

        # Forward-fill 4H regime signals + raw score to 15m timestamps (no lookahead)
        regime_4h = pd.DataFrame({"bull": bull_4h, "bear": bear_4h, "score": trend_score_4h})
        regime_15m = regime_4h.reindex(out.index, method="ffill")
        macro_up = regime_15m["bull"].fillna(False)
        macro_dn = regime_15m["bear"].fillna(False)
        out["h4_regime_score_w"] = regime_15m["score"].fillna(0)

    # Optional: gate on 4h MACD histogram slope — blocks fading crossovers.
    # Fires when ema_fast/slow are 12/26 (MACD periods) or any config with noise risk.
    if p.get("macd_slope_gate", False):
        macd4      = ef4 - es4
        hist_rising  = macd4 > macd4.shift(1)
        hist_falling = macd4 < macd4.shift(1)
        macro_up = macro_up & hist_rising
        macro_dn = macro_dn & hist_falling

    # ── Layer 2: 1H mid trend ─────────────────────────────────────────────────
    ef1, es1 = _ema(out["h1_close"], p["ema_fast"]), _ema(out["h1_close"], p["ema_slow"])
    mid_up, mid_dn = _htf_dir(ef1, es1, out["h1_close"], htf_mode, slope_bars, sep_guard, slope_pct)

    # ── HTF Momentum Score (opt-in; replaces binary Layer 1 + 2 gates) ────────
    # 8 components: direction + separation + price-confirm + RSI on each of 4h/1h.
    # Require min_htf_score/8. Default 6/8 filters noise crossovers and weak trends.
    if p.get("htf_mom_score", False):
        sep_th4       = p.get("htf_sep_thresh_4h", 0.002)
        sep_th1       = p.get("htf_sep_thresh_1h", 0.001)
        rsi_bull_lo   = p.get("htf_rsi_bull_lo",  45.0)
        rsi_bull_hi   = p.get("htf_rsi_bull_hi",  75.0)
        rsi_bear_lo   = p.get("htf_rsi_bear_lo",  25.0)
        rsi_bear_hi   = p.get("htf_rsi_bear_hi",  55.0)
        min_htf       = p.get("min_htf_score",      6)

        rsi4 = _rsi(out["h4_close"], p["rsi_period"])
        sep4 = (ef4 - es4).abs() / out["h4_close"].replace(0, np.nan)

        rsi1 = _rsi(out["h1_close"], p["rsi_period"])
        sep1 = (ef1 - es1).abs() / out["h1_close"].replace(0, np.nan)

        htf_long = (
            (ef4 > es4).astype(float)                                  # 4h direction
            + (sep4 >= sep_th4).astype(float)                          # 4h separation
            + (out["h4_close"] > ef4).astype(float)                    # 4h px confirm
            + ((rsi4 >= rsi_bull_lo) & (rsi4 <= rsi_bull_hi)).astype(float)   # 4h RSI
            + (ef1 > es1).astype(float)                                # 1h direction
            + (sep1 >= sep_th1).astype(float)                          # 1h separation
            + (out["h1_close"] > ef1).astype(float)                    # 1h px confirm
            + ((rsi1 >= rsi_bull_lo) & (rsi1 <= rsi_bull_hi)).astype(float)   # 1h RSI
        )
        htf_short = (
            (ef4 < es4).astype(float)
            + (sep4 >= sep_th4).astype(float)
            + (out["h4_close"] < ef4).astype(float)
            + ((rsi4 >= rsi_bear_lo) & (rsi4 <= rsi_bear_hi)).astype(float)
            + (ef1 < es1).astype(float)
            + (sep1 >= sep_th1).astype(float)
            + (out["h1_close"] < ef1).astype(float)
            + ((rsi1 >= rsi_bear_lo) & (rsi1 <= rsi_bear_hi)).astype(float)
        )
        out["htf_score_long"]  = htf_long
        out["htf_score_short"] = htf_short
        macro_up = htf_long  >= min_htf
        macro_dn = htf_short >= min_htf
        mid_up   = macro_up   # 1h already baked into score; pass AND gate downstream
        mid_dn   = macro_dn

    # ── HTF Stability: require N×15m of consistent direction (anti-false-flip) ──
    # Prevents "early" mode from triggering on a single-bar HTF spike.
    # 2 bars = 30 min; 4 bars = 60 min. Default 2 catches the XAG-style fast reversal.
    _stab = int(p.get("htf_stability_bars", 2))
    if _stab > 1:
        macro_up = macro_up.rolling(_stab, min_periods=_stab).min().fillna(False).astype(bool)
        macro_dn = macro_dn.rolling(_stab, min_periods=_stab).min().fillna(False).astype(bool)
        mid_up   = mid_up.rolling(_stab, min_periods=_stab).min().fillna(False).astype(bool)
        mid_dn   = mid_dn.rolling(_stab, min_periods=_stab).min().fillna(False).astype(bool)

    # ── Layer 3: Pullback zone (mode-specific) ────────────────────────────────
    if fast_mode:
        # Fast Mode v2: 1H EMA20 zone (or HMA20 if use_hma20_pullback=True)
        if p.get("use_hma20_pullback", False):
            # Compute on actual 1H bars then reindex — avoids ffill HMA distortion
            ema20_1h = _hma(df1h["close"], 20).reindex(out.index, method="ffill")
        else:
            ema20_1h = ef1
        # Pullback proximity reference. Default uses the last CLOSED 1H bar
        # (h1_close), which is stale for up to ~1h — an intra-hour dip-and-bounce
        # to the 1H EMA20 isn't seen until the 1H bar closes near it, so a fresh
        # pullback entry can lag up to 45 min. pullback_live_15m=True compares the
        # LIVE 15m close/low/high against the 1H EMA20 instead, re-evaluating the
        # zone every 15m so intra-hour touches are caught immediately.
        _live = bool(p.get("pullback_live_15m", False))
        _px   = out["close"] if _live else out["h1_close"]
        _lo   = out["low"]   if _live else out["h1_low"]
        _hi   = out["high"]  if _live else out["h1_high"]
        pullback_atr_mult = float(p.get("pullback_atr_mult_fast", 1.2))
        if pullback_atr_mult > 0:
            # ATR-based: adaptive to volatility (BTC 1H ATR ~$600-1k; 1.2× ~ $720-1.2k)
            h1_atr = _atr(df1h, 14).shift(1)  # shift 1 bar: no-lookahead
            atr_1h = h1_atr.reindex(out.index, method="ffill").bfill()
            near_ema = (_px - ema20_1h).abs() <= atr_1h * pullback_atr_mult
        else:
            pullback_pct = p["pullback_pct_fast"]
            near_ema = (_px - ema20_1h).abs() / ema20_1h <= pullback_pct
        wick_bounce_l = (_lo <= ema20_1h * 1.003) & (_px > ema20_1h)
        wick_reject_s = (_hi >= ema20_1h * 0.997) & (_px < ema20_1h)
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
    comp_pct = _mtf_bias(out["close"], {"1h": out["h1_close"], "4h": out["h4_close"]},
                         include_primary=not bool(p.get("bias_htf_only", False)))
    out["comp_pct"] = comp_pct
    long_bias_ok  = comp_pct > bias_gate
    short_bias_ok = comp_pct < -bias_gate

    # ── Layer 5: ADX gate (mode-specific) ────────────────────────────────────
    out["atr"]   = _atr(out, p["atr_period"])
    out["adx15"] = _adx(out, p["adx_len"])
    if fast_mode:
        # ADX in (adx_min_fast, adx_max_fast] — active but not overheated.
        adx_ok = ((out["adx15"] > p["adx_min_fast"])
                  & (out["adx15"] <= p["adx_max_fast"]))
        if p.get("adx_rising_fast", True):   # honor TCI_ADX_RISING toggle
            adx_rising = out["adx15"] > out["adx15"].shift(1).fillna(0)
            # Optional relax: accept "rising OR already-strong" so a pullback that
            # keeps ADX high (but flat/dipping) still qualifies — enters 1-2 bars sooner.
            strong_th = p.get("adx_rising_or_strong", 0)
            if strong_th and strong_th > 0:
                adx_rising = adx_rising | (out["adx15"] > strong_th)
            adx_ok = adx_ok & adx_rising
    else:
        adx_ok = out["adx15"] > p["adx_min"]

    # Optional: require 1h AND 4h ADX to confirm trend is active on higher TFs.
    if p.get("htf_adx_gate", False):
        adx_len  = p.get("htf_adx_len", 14)
        min_1h   = p.get("htf_adx_min_1h", 20)
        min_4h   = p.get("htf_adx_min_4h", 18)
        adx1h = _adx(out[["h1_high","h1_low","h1_close"]].rename(
            columns={"h1_high":"high","h1_low":"low","h1_close":"close"}), adx_len)
        adx4h = _adx(out[["h4_high","h4_low","h4_close"]].rename(
            columns={"h4_high":"high","h4_low":"low","h4_close":"close"}), adx_len)
        adx_ok = adx_ok & (adx1h > min_1h) & (adx4h > min_4h)

    # ATR Percentile — rolling rank (0–100); <20=quiet, >80=volatile.
    # Computed unconditionally: used by regime + exposed as monitoring metric.
    _atr_pct_win = int(p.get("regime_atr_pct_window", 100))
    atr_pct = _atr_percentile(out["atr"], _atr_pct_win)
    out["atr_pct"] = atr_pct

    # Chop Index on 1H — always computed (used by regime + optional chop gate).
    _chop_1h = _chop_index(df1h, 14).reindex(out.index, method="ffill").fillna(50.0)
    out["chop_1h"] = _chop_1h

    # ── Layer 6: Micro score indicators ──────────────────────────────────────
    ema9  = _ema(out["close"], p["ema_micro"])
    ema20 = _ema(out["close"], p["ema_fast"])
    rsi15 = _rsi(out["close"], p["rsi_period"])
    volma = _sma(out["volume"], p["vol_period"])
    vol_ok = (volma > 0) & (out["volume"] >= volma * p["vol_mult"])
    vol_expansion_ok = (volma > 0) & (out["volume"] >= volma * 1.15)
    rel_vol = (out["volume"] / volma.replace(0, np.nan)).fillna(1.0)
    out["rel_vol"] = rel_vol

    # ATR compression: ATR14 was tighter than ATR50 (squeeze), now expanding
    atr50_15m = _rma(_true_range(out), 50)
    atr_was_compressed = (
        (out["atr"].shift(1) < atr50_15m.shift(1)) |
        (out["atr"].shift(2) < atr50_15m.shift(2)) |
        (out["atr"].shift(3) < atr50_15m.shift(3))
    )
    atr_expanding_now = out["atr"] > out["atr"].shift(1)
    atr_compress_ok = atr_was_compressed & atr_expanding_now

    macd_line = _ema(out["close"], 12) - _ema(out["close"], 26)
    macd_sig  = _ema(macd_line, 9)

    # SJ Hybrid scoring extras (only computed when sj_scoring=True)
    hma_period = int(p.get("hma_period", 20))
    ema5  = _ema(out["close"], 5)
    sma9  = _sma(out["close"], 9)
    hma   = _hma(out["close"], hma_period)
    bk_lb = int(p.get("breakout_lookback", 10))
    hh10  = out["high"].rolling(bk_lb).max().shift(1)
    ll10  = out["low"].rolling(bk_lb).min().shift(1)
    # BOS: Break of Structure — close breaks beyond N-bar swing high/low
    bos_lb   = int(p.get("bos_lookback", 5))
    bos_hh   = out["high"].rolling(bos_lb).max().shift(1)
    bos_ll   = out["low"].rolling(bos_lb).min().shift(1)
    bos_bull = out["close"] > bos_hh   # bullish BOS: break above N-bar high
    bos_bear = out["close"] < bos_ll   # bearish BOS: break below N-bar low
    # Prior BOS: structural break happened 1-20 bars ago — trend established before current pullback
    prior_bos_bull = bos_bull.shift(1).rolling(20).max().fillna(0).astype(bool)
    prior_bos_bear = bos_bear.shift(1).rolling(20).max().fillna(0).astype(bool)

    # ZLEMA: zero-lag EMA (always computed — cheap, used by sj_zlema component)
    zlema9_15m  = _zlema(out["close"], 9)
    zlema20_15m = _zlema(out["close"], 20)

    # Price Action: pin bar / engulfing (used by sj_price_action component, no lookahead)
    _body     = (out["close"] - out["open"]).abs()
    _wick_bot = (out[["open", "close"]].min(axis=1) - out["low"]).clip(lower=0)
    _wick_top = (out["high"] - out[["open", "close"]].max(axis=1)).clip(lower=0)
    _pin_bull = (_wick_bot > _body * 2) & (out["close"] >= out["open"])
    _pin_bear = (_wick_top > _body * 2) & (out["close"] <= out["open"])
    _eng_bull = (
        (out["close"] > out["open"]) &
        (out["close"] > out["open"].shift(1)) &
        (out["open"]  < out["close"].shift(1))
    )
    _eng_bear = (
        (out["close"] < out["open"]) &
        (out["close"] < out["open"].shift(1)) &
        (out["open"]  > out["close"].shift(1))
    )
    pa_bull = (_pin_bull | _eng_bull).fillna(False)
    pa_bear = (_pin_bear | _eng_bear).fillna(False)

    # RSI Hidden Divergence (shift-based, no lookahead, no center=True)
    # Hidden bull: price higher than N bars ago but RSI lower → bullish continuation
    # Hidden bear: price lower than N bars ago but RSI higher → bearish continuation
    _rdiv_lb        = int(p.get("rsi_div_lookback", 5))
    _price_chg      = out["close"] - out["close"].shift(_rdiv_lb)
    _rsi_chg        = rsi15 - rsi15.shift(_rdiv_lb)
    hidden_bull_div = ((_price_chg > 0) & (_rsi_chg < 0)).fillna(False)
    hidden_bear_div = ((_price_chg < 0) & (_rsi_chg > 0)).fillna(False)

    # OBV trend: cumulative volume flow vs its own EMA20
    obv_dir  = np.sign(out["close"].diff().fillna(0))
    obv      = (obv_dir * out["volume"]).cumsum()
    obv_ema20 = _ema(obv, 20)
    # SJ Extended extras: market structure + FVG imbalance
    sma20         = _sma(out["close"], 20)
    sma20_rising  = sma20 > sma20.shift(3)              # slope over 3 bars (45 min)
    # FVG: 3-candle imbalance where there's a gap between bar[i-2].high and bar[i].low
    fvg_bull = (out["low"] > out["high"].shift(2)).rolling(5, min_periods=1).max().astype(bool)
    fvg_bear = (out["high"] < out["low"].shift(2)).rolling(5, min_periods=1).max().astype(bool)
    roc9_val = _roc(out["close"], 9)

    # Confidence Score extras — EMA slope, ATR expansion, ADX rising (score components)
    ema50         = _ema(out["close"], 50)
    # EMA slope: both EMA20 AND EMA50 rising over 2 bars → structural trend confirmed
    ema_slope_ok  = (ema20 > ema20.shift(2)) & (ema50 > ema50.shift(2))
    # ATR expansion: ATR14 above its 20-bar MA → market is moving, not in quiet consolidation
    atr_ma20      = _sma(out["atr"], 20)
    atr_expand_ok = out["atr"] > atr_ma20
    # ADX rising (2-bar lookback): trend is strengthening steadily, not one-bar spike
    adx_rising_ok = out["adx15"] > out["adx15"].shift(2).fillna(0)

    ctx = dict(
        close=out["close"], open=out["open"], ema9=ema9, ema20=ema20, rsi15=rsi15, vol_ok=vol_ok,
        macd=macd_line, macd_signal=macd_sig,
        rel_vol=rel_vol, rel_vol_min=float(p.get("rel_vol_min", 1.2)),
        ema5=ema5, sma9=sma9, hma=hma, hh10=hh10, ll10=ll10,
        obv=obv, obv_ema20=obv_ema20,
        sma20=sma20, sma20_rising=sma20_rising,
        fvg_bull=fvg_bull, fvg_bear=fvg_bear,
        roc9=roc9_val,
        vol_expansion_ok=vol_expansion_ok,
        atr_compress_ok=atr_compress_ok,
        bos_bull=bos_bull, bos_bear=bos_bear,
        prior_bos_bull=prior_bos_bull, prior_bos_bear=prior_bos_bear,
        zlema9=zlema9_15m, zlema20=zlema20_15m,
        pa_bull=pa_bull, pa_bear=pa_bear,
        hidden_bull_div=hidden_bull_div, hidden_bear_div=hidden_bear_div,
        rsi_min_buy=p["rsi_min_buy"], rsi_max_buy=p["rsi_max_buy"],
        rsi_min_sell=p["rsi_min_sell"], rsi_max_sell=p["rsi_max_sell"],
        ema_slope_ok=ema_slope_ok,
        atr_expand_ok=atr_expand_ok,
        adx_rising_ok=adx_rising_ok,
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

    # ── Market Regime classification ──────────────────────────────────────────
    _regime_sl_mult   = pd.Series(1.0, index=out.index)
    _regime_entry_ok  = pd.Series(True, index=out.index)  # diagnostic default
    if p.get("market_regime_enabled", True):
        _regime, _regime_entry_ok, _regime_sl_mult = _market_regime(
            out["adx15"], _chop_1h, atr_pct, ema20, rsi15, out["close"], p
        )
        out["regime"] = _regime
        out["regime_sl_mult"] = _regime_sl_mult

    # ── ATR compression gate (optional, separate from SJ scoring) ────────────
    if p.get("atr_compress_gate", False):
        long_gates  = adx_ok & atr_compress_ok
        short_gates = adx_ok & atr_compress_ok
    else:
        long_gates  = adx_ok
        short_gates = adx_ok

    # ── Market Regime gate: block RANGE / LOW_VOL / EXHAUSTION entries ────────
    if p.get("market_regime_enabled", True):
        long_gates  = long_gates  & _regime_entry_ok
        short_gates = short_gates & _regime_entry_ok

    # ── Chop Index gate: block entries in ranging/sideways 1H market ─────────
    _chop_ok_diag = pd.Series(True, index=out.index)  # diagnostic default
    if p.get("chop_filter_enabled", False):
        _chop_thresh = float(p.get("chop_threshold", 50.0))
        _chop_ok = _chop_1h < _chop_thresh   # LOW = trending; HIGH = choppy
        _chop_ok_diag = _chop_ok
        long_gates  = long_gates  & _chop_ok
        short_gates = short_gates & _chop_ok

    # ── Volume boost gate: entry bar must have elevated volume ────────────────
    if p.get("vol_boost_gate", False):
        _vb_mult = float(p.get("vol_boost_mult", 1.5))
        _vol_boost_ok = (volma > 0) & (out["volume"] >= volma * _vb_mult)
        long_gates  = long_gates  & _vol_boost_ok
        short_gates = short_gates & _vol_boost_ok

    # ── Pressure / squeeze-expansion detection (only when "pressure" mode) ────
    # A "build-up then push" pattern: the prior 1-2 bars coil (contracting range),
    # then the current bar EXPANDS with a strong directional close — the momentum-
    # ignition bar. The intuition was that entering on this bar gives a better
    # average price and fewer stop-outs. Backtest Jan-May 2026 (BTC+XAG+XAU),
    # train/test split, DISPROVED it: vs relaxed (TRAIN PF 1.95 / TEST PF 1.33),
    # pressure got TRAIN PF 1.88 / TEST PF 1.18 with SL rate UP (18.0%→20.5%) and
    # 60% fewer trades. Buying the expansion bar often buys the top OF the
    # expansion, not the start of the move. Kept off; left for reference.
    _trig_mode = p.get("final_trigger_mode") or ("strict" if p.get("final_trigger_enabled", False) else "off")
    pressure_long  = pd.Series(False, index=out.index)
    pressure_short = pd.Series(False, index=out.index)
    if _trig_mode == "pressure":
        _rng      = (out["high"] - out["low"]).replace(0, np.nan)
        _body_up  = (out["close"] - out["low"])  / _rng   # 1.0 = closed on the high
        _body_dn  = (out["high"] - out["close"]) / _rng   # 1.0 = closed on the low
        _pr_ratio = float(p.get("pressure_body_ratio", 0.60))
        _buildup  = (_rng.shift(1) < _rng.shift(2)) | (_rng.shift(1) < out["atr"].shift(1))
        _expand   = _rng > _rng.shift(1)
        pressure_long  = ((out["close"] > out["open"]) & (_body_up >= _pr_ratio)
                          & _expand & _buildup & (out["close"] > out["close"].shift(1))).fillna(False)
        pressure_short = ((out["close"] < out["open"]) & (_body_dn >= _pr_ratio)
                          & _expand & _buildup & (out["close"] < out["close"].shift(1))).fillna(False)

    # ── Momentum confirmation gate: enter early, not after peak ──────────────
    # Blocks entering after momentum has already peaked. momentum_gate_mode picks
    # HOW momentum is confirmed — the laggiest piece of the entry stack, so the
    # choice directly controls entry timing:
    #   "macd"   MACD histogram rising over N bars (default; laggy — EMA12/26/9)
    #   "volmom" price up AND volume >= MA20 (participation) — reacts in 1 bar,
    #            the volume filter replaces MACD's smoothing as the false-move guard
    #   "roc3"   ROC(3) favorable — fast raw momentum, no volume
    #   "volmom_macd" volmom OR macd — either confirms (loosest)
    _mgate = p.get("momentum_gate_mode", "macd")
    _macd_hist_l_diag = pd.Series(True, index=out.index)  # diagnostic default
    _macd_hist_s_diag = pd.Series(True, index=out.index)
    if p.get("macd_hist_rising_gate", True):
        macd_hist       = macd_line - macd_sig
        _mh_lb          = int(p.get("macd_hist_lookback", 2))
        _macd_l         = macd_hist > macd_hist.shift(_mh_lb)
        _macd_s         = macd_hist < macd_hist.shift(_mh_lb)
        # volume-backed momentum tick: price rises AND the bar carries >= average
        # volume — genuine participation, not a low-volume drift. Reacts in 1 bar.
        _vol_ok_g       = out["volume"] >= volma
        _vm_l           = (out["close"] > out["close"].shift(1)) & _vol_ok_g
        _vm_s           = (out["close"] < out["close"].shift(1)) & _vol_ok_g
        _roc3           = _roc(out["close"], 3)
        _r3_l, _r3_s    = _roc3 > 0, _roc3 < 0
        if   _mgate == "volmom":      mom_l, mom_s = _vm_l, _vm_s
        elif _mgate == "roc3":        mom_l, mom_s = _r3_l, _r3_s
        elif _mgate == "volmom_macd": mom_l, mom_s = (_vm_l | _macd_l), (_vm_s | _macd_s)
        else:                          mom_l, mom_s = _macd_l, _macd_s   # "macd"
        if _trig_mode == "pressure":
            mom_l = mom_l | pressure_long
            mom_s = mom_s | pressure_short
        _macd_hist_l_diag = mom_l
        _macd_hist_s_diag = mom_s
        long_gates  = long_gates  & mom_l
        short_gates = short_gates & mom_s

    # ── MACD-histogram peak/exhaustion filter (opt-in) ───────────────────────
    # Blocks "buying the top": the price bar can tick up with volume (passing
    # volmom) while the MACD histogram has already PEAKED and is rolling over —
    # a weak, exhausting bounce (the classic ETH "entered at the top" loss).
    # Detects a post-peak decline: the histogram fell for `slope_bars` and a
    # meaningfully higher peak existed within the last `lookback` bars.
    if p.get("macd_peak_filter", False):
        _h        = macd_line - macd_sig
        _pk_lb    = int(p.get("macd_peak_lookback", 6))
        _pk_sb    = int(p.get("macd_peak_slope_bars", 2))
        _decl_l   = _h < _h.shift(_pk_sb)                 # histogram sloping down
        _peak_l   = _h.rolling(_pk_lb).max() > _h         # a higher peak exists behind us
        _post_peak_l = (_decl_l & _peak_l).fillna(False)  # rolling over from a recent top
        _incl_s   = _h > _h.shift(_pk_sb)                 # histogram sloping up (short exhausting)
        _trough_s = _h.rolling(_pk_lb).min() < _h
        _post_peak_s = (_incl_s & _trough_s).fillna(False)
        long_gates  = long_gates  & ~_post_peak_l
        short_gates = short_gates & ~_post_peak_s

    # ── Final entry trigger: momentum confirmation before entry ──────────────
    # Diagnostic (gate_bottleneck): this is the #1 entry blocker (804 otherwise-
    # complete setups). "strict" waits for a break of the prior bar's high/low —
    # accurate but enters late at a worse price. "relaxed" only needs the close to
    # rise over the prior close (a softer momentum tick) — enters ~1 bar sooner.
    # "pressure" fires on the squeeze→expansion ignition bar (build-up then push).
    # False disables it entirely (earliest, most trades).
    if _trig_mode == "strict":
        trigger_long  = out["close"] > out["high"].shift(1)
        trigger_short = out["close"] < out["low"].shift(1)
    elif _trig_mode == "relaxed":
        trigger_long  = out["close"] > out["close"].shift(1)
        trigger_short = out["close"] < out["close"].shift(1)
    elif _trig_mode == "pressure":
        trigger_long  = pressure_long
        trigger_short = pressure_short
    else:  # "off"
        trigger_long  = pd.Series(True, index=out.index)
        trigger_short = pd.Series(True, index=out.index)

    # ── Simple Fast Entry: replace all inner gates with HMA direction + MACD hist sign ──
    # Replaces: pullback zone / chop / regime / MACD-rising / final-trigger with just:
    #   long  = close > HMA16  AND  MACD_hist > 0  AND  ADX > min
    #   short = close < HMA16  AND  MACD_hist < 0  AND  ADX > min
    # Backtest Jan-May 2026 (BTC+XAG+XAU): letting score alone decide (ADX-only gate)
    # produced 452 trades but PF 0.72 (losing) — HMA+MACD hard-gate is the real anchor,
    # not a redundant duplicate of the score; the score is a secondary quality filter
    # on TOP of the direction/momentum gate, not a replacement for it.
    # Keeps: 4H macro, 1H mid, MTF bias, score ≥ min_score (safety guards unchanged).
    if fast_mode and p.get("simple_fast_entry", False):
        _sf_hist      = macd_line - macd_sig
        long_gates    = adx_ok & (out["close"] > hma) & (_sf_hist > 0)
        short_gates   = adx_ok & (out["close"] < hma) & (_sf_hist < 0)
        at_pull_long  = pd.Series(True, index=out.index)
        at_pull_short = pd.Series(True, index=out.index)
        trigger_long  = pd.Series(True, index=out.index)
        trigger_short = pd.Series(True, index=out.index)

    # ── Score-Primary Entry (experimental, separate from simple_fast_entry) ──
    # ADX is the only hard inner gate; direction/momentum/structure fully
    # delegated to the weighted score ≥ min_score. Kept as a DISTINCT flag from
    # simple_fast_entry (which stays the safe hard-gate default) so this can be
    # A/B tested at different min_score thresholds without touching production.
    if fast_mode and p.get("score_primary_entry", False):
        long_gates    = adx_ok
        short_gates   = adx_ok
        at_pull_long  = pd.Series(True, index=out.index)
        at_pull_short = pd.Series(True, index=out.index)
        trigger_long  = pd.Series(True, index=out.index)
        trigger_short = pd.Series(True, index=out.index)

    # ── Regime-adaptive min_score (opt-in) ────────────────────────────────────
    # STRONG_TREND is the highest-WR environment — demanding the same score
    # there as in marginal TREND wastes speed; HIGH_VOL is the riskiest allowed
    # regime — demanding more confirmation there cuts false signals.
    _ms = pd.Series(float(p["min_score"]), index=out.index)
    if (p.get("regime_adaptive_min_score", False)
            and p.get("market_regime_enabled", True) and "regime" in out):
        _ms[out["regime"] == "STRONG_TREND"] = float(p.get("min_score_strong",  50.0))
        _ms[out["regime"] == "HIGH_VOL"]     = float(p.get("min_score_highvol", 70.0))

    # ── Strong-trend breakout entry (opt-in): second entry path ──────────────
    # The pullback zone is the only entry path — in a strong trend price often
    # never returns to the 1H zone and the whole run is missed. In STRONG_TREND
    # with an exceptionally high score (default ≥80), waive the pullback
    # requirement only. Unlike score_primary_entry (which waived every gate in
    # every regime and lost badly), this waives ONE gate in the single best
    # regime under a much higher score bar; all other gates still apply.
    if (p.get("strong_trend_entry", False)
            and p.get("market_regime_enabled", True) and "regime" in out):
        _st_ok = out["regime"] == "STRONG_TREND"
        _st_sc = float(p.get("strong_trend_score", 80.0))
        # Track when the waiver is the DECIDING factor (pullback itself failed) so
        # the signal metadata can label the entry path for the lesson tracker.
        _stw_l = _st_ok & (score_long  >= _st_sc) & ~at_pull_long.fillna(False).astype(bool)
        _stw_s = _st_ok & (score_short >= _st_sc) & ~at_pull_short.fillna(False).astype(bool)
        out["d_stw_l"] = _stw_l
        out["d_stw_s"] = _stw_s
        at_pull_long  = at_pull_long  | _stw_l
        at_pull_short = at_pull_short | _stw_s

    out["final_buy"]  = (macro_up & mid_up  & at_pull_long  & long_bias_ok  &
                          (score_long  >= _ms) & long_gates  & trigger_long).fillna(False)
    out["final_sell"] = (macro_dn  & mid_dn  & at_pull_short & short_bias_ok &
                          (score_short >= _ms) & short_gates & trigger_short).fillna(False)

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
    # Inner gate diagnostics (regime/chop are symmetric; macd/trigger are directional)
    out["d_regime_ok"] = _regime_entry_ok.fillna(True)
    out["d_chop_ok"]   = _chop_ok_diag.fillna(True)
    out["d_macd_l"]    = _macd_hist_l_diag.fillna(True)
    out["d_macd_s"]    = _macd_hist_s_diag.fillna(True)
    out["d_trig_l"]    = trigger_long.fillna(True)
    out["d_trig_s"]    = trigger_short.fillna(True)

    # ── Dynamic SL: ATR14/ATR50 ratio adjusts sl_mult to market volatility ──────
    if p.get("dynamic_sl_enabled", False):
        _atr50_sl   = _rma(_true_range(out), 50)
        _atr_ratio  = (out["atr"] / _atr50_sl.replace(0, np.nan)).fillna(1.0)
        _dyn_hi     = float(p.get("dynamic_sl_high_mult", 1.3))
        _dyn_lo     = float(p.get("dynamic_sl_low_mult",  0.8))
        _base_mult  = pd.Series(
            np.where(_atr_ratio > 1.3, p["sl_mult"] * _dyn_hi,
            np.where(_atr_ratio < 0.8, p["sl_mult"] * _dyn_lo,
                     p["sl_mult"])),
            index=out.index)
    else:
        _base_mult = pd.Series(p["sl_mult"], index=out.index)

    # Regime SL adjustment: HIGH_VOL widens stop to avoid premature shake-out.
    _final_mult = _base_mult * _regime_sl_mult
    dist = (out["atr"] * _final_mult).clip(
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


# ── REVERSAL SPIKE-CUT ────────────────────────────────────────────────────────

def check_spike_cut(side: str, entry: float, one_r: float,
                    df3m: pd.DataFrame,
                    spike_atr_mult: float = 1.5,
                    spike_bars: int = 4,
                    uw_frac: float = 0.0) -> tuple[str, str]:
    """
    Fast adverse-move detector using 3m candles.
    Fires when price moves ≥ spike_atr_mult×ATR(3m) against the position
    within spike_bars×3m (default 4×3m = 12 min) AND the position is at least
    uw_frac×R underwater (default 0.0 = no floor, legacy behaviour).

    uw_frac exists because ATR(3m) shrinks in quiet markets, so a fixed
    ATR-multiple threshold can trip on ordinary 3-minute noise that never
    puts the position at real risk — backtest Jan-May 2026 (BTC+XAU) showed
    uw_frac=0.0 cutting 82% of trades at 0.1-0.14R underwater (near noise
    level), taking a PF-1.81 entry edge to PF-0.41. A floor keeps spike-cut
    as a genuine fast-bailout for positions already meaningfully underwater,
    not a rare-event detector, without moving the ATR threshold.
    """
    min_bars = spike_bars + 15          # need enough history for ATR(14)
    if df3m is None or len(df3m) < min_bars:
        return ("HOLD", "insufficient 3m data for spike-cut")

    atr3 = float(_atr(df3m, 14).iloc[-1])
    if atr3 <= 0:
        return ("HOLD", "invalid ATR(3m) — spike-cut skipped")

    px_now  = float(df3m["close"].iloc[-1])
    px_prev = float(df3m["close"].iloc[-(spike_bars + 1)])

    uw = ((entry - px_now) / one_r if side == "long"
          else (px_now - entry) / one_r)
    if uw < uw_frac:
        return ("HOLD", f"only {uw:.2f}R underwater (< {uw_frac}R) — spike-cut floor not met")

    # Adverse move: down for long, up for short
    adverse_move = (px_prev - px_now) if side == "long" else (px_now - px_prev)
    move_r = adverse_move / atr3

    if move_r >= spike_atr_mult:
        return ("CLOSE",
                f"SPIKE-CUT: {move_r:.1f}×ATR(3m) adverse in {spike_bars}bars "
                f"(uw={uw:.2f}R, atr={atr3:.2f})")
    return ("HOLD",
            f"spike {move_r:.2f}×ATR < {spike_atr_mult}×ATR — safe")


# ── TREND-FADE CUT ─────────────────────────────────────────────────────────────

def check_trend_fade(side: str, entry: float, one_r: float,
                     df15: pd.DataFrame, df5: pd.DataFrame,
                     uw_frac: float = 0.6) -> tuple[str, str]:
    """
    Early-loss cut for a dying trend. Fires ONLY when ALL three are true:
      1. ADX(15m) is falling     — trend strength weakening
      2. price has lost EMA20(5m) against the position — direction turned adverse
      3. position is ≥ uw_frac×1R underwater — the trade is genuinely losing

    Cuts the loser at ~−0.6R instead of waiting for the full −1R stop. The triple
    gate is what makes it safe: a quiet winner (not underwater) or a healthy
    pullback (price still above EMA20, or ADX rising) is never touched — that is
    why ADX-alone exits failed in backtest while this one improves both PnL and
    drawdown.

    Backtest Jan-May 2026 ($50×20x), combined BTC+XAU vs current (BE-only):
      Classic:   $241 → $279 (+16%), MaxDD −$108 → −$105, PnL/DD 2.23 → 2.66
      SJ Hybrid: $143 → $217 (+52%), MaxDD −$172 → −$123, PnL/DD 0.83 → 1.76
    """
    if df15 is None or len(df15) < 30 or df5 is None or len(df5) < 25:
        return ("HOLD", "insufficient data for trend-fade")
    if one_r <= 0:
        return ("HOLD", "invalid one_r — trend-fade skipped")

    px = float(df5["close"].iloc[-1])
    uw = (entry - px) / one_r if side == "long" else (px - entry) / one_r
    if uw < uw_frac:
        return ("HOLD", f"only {uw:.2f}R underwater (< {uw_frac}R) — safe")

    # Use the last two CLOSED 15m bars (iloc[-1] is still forming) so the
    # "ADX falling" read is stable and matches the backtest that validated this.
    adx15 = _adx(df15, 14)
    adx_now, adx_prev = float(adx15.iloc[-2]), float(adx15.iloc[-3])
    adx_falling = adx_now < adx_prev

    ema20_5 = float(_ema(df5["close"], 20).iloc[-1])
    against = (px < ema20_5) if side == "long" else (px > ema20_5)

    if adx_falling and against:
        return ("CLOSE",
                f"TREND-FADE: {uw:.2f}R underwater + ADX(15m) falling "
                f"({adx_prev:.0f}→{adx_now:.0f}) + lost EMA20(5m)")
    return ("HOLD",
            f"{uw:.2f}R underwater but trend not fading "
            f"(adx {adx_prev:.0f}→{adx_now:.0f}, against={against})")


# ── REVERSAL-CUT (grinding V-reversal, no HTF/ADX-falling required) ──────────

def check_reversal_cut(side: str, entry: float, one_r: float,
                       df5: pd.DataFrame, uw_frac: float = 0.7) -> tuple[str, str]:
    """
    Local-momentum reversal cut. Catches a grinding V-reversal that the other
    two guards structurally miss:
      - crash-guard needs HTF (1h/4h) composite bias to fully flip past ±65 —
        HTF lags a fresh 5m reversal by design.
      - trend-fade needs ADX(15m) to be FALLING — but a genuine reversal often
        makes ADX RISE (a new trend is forming against the position), so the
        gate never fires.
      - spike-cut needs a fast move (≥1.5×ATR(3m) in 12 min) — misses a
        slower grind-up/down over 30-60+ min that still crosses ~1R.

    Fires when ALL of:
      1. position is >= uw_frac R underwater
      2. price is beyond EMA9(5m) against the position
      3. EMA9(5m) slope is ACCELERATING against the position (latest 2-bar
         slope steeper than the prior 2-bar slope) — confirms sustained
         momentum, not a single noisy tick.
    """
    if df5 is None or len(df5) < 10:
        return ("HOLD", "insufficient 5m data for reversal-cut")
    if one_r <= 0:
        return ("HOLD", "invalid one_r — reversal-cut skipped")

    px = float(df5["close"].iloc[-1])
    uw = (entry - px) / one_r if side == "long" else (px - entry) / one_r
    if uw < uw_frac:
        return ("HOLD", f"only {uw:.2f}R underwater (< {uw_frac}R) — safe")

    ema9 = _ema(df5["close"], 9)
    e9_now     = float(ema9.iloc[-1])
    slope_now  = float(ema9.iloc[-1] - ema9.iloc[-3])
    slope_prev = float(ema9.iloc[-3] - ema9.iloc[-5])

    if side == "long":
        against_px   = px < e9_now
        accelerating = (slope_now < 0) and (slope_now < slope_prev)
    else:
        against_px   = px > e9_now
        accelerating = (slope_now > 0) and (slope_now > slope_prev)

    if against_px and accelerating:
        return ("CLOSE",
                f"REVERSAL-CUT: {uw:.2f}R underwater + EMA9(5m) accelerating against "
                f"(slope {slope_prev:.2f}→{slope_now:.2f})")
    return ("HOLD",
            f"{uw:.2f}R underwater but momentum not confirmed accelerating against "
            f"(against_px={against_px}, accelerating={accelerating})")


# ── BaseStrategy wrapper ─────────────────────────────────────────────────────

class TrendContImprovedStrategy(BaseStrategy):
    """
    TrendContinuation Improved v2: 15m primary + 1h/4h MTF.
    STRICT: 15m swing pullback + ADX>30, TP2=2.5R.
    FAST v2: 1H EMA20 ±1.8% + bias>60 + ADX>18-rising + cooldown, TP2=3.0R.
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
        tp1_r=0.5, tp1_fraction=0.50, tp2_r=2.5,   # close 50% at TP1 (keep a bigger runner)
        min_score=60.0,  # 100-point confidence scale: entry gate ≥60, sizing: ≥90 strong, ≥70 normal, <70 small(0.5x)
        # Fast mode v2
        fast_mode=False,
        adx_min_fast=15,          # ADX lower bound — trend must be active. Lowered 18→15
                                  # (keep adx_rising_fast=True) to enter 1-2 bars earlier on
                                  # a fresh trend; the "rising" gate still blocks dead chop.
        adx_max_fast=45,          # ADX upper bound — caps overheated entries.
        adx_rising_fast=True,     # also require ADX[0] > ADX[1]
        pullback_pct_fast=0.025,  # fallback pct zone (used when pullback_atr_mult_fast=0)
        pullback_atr_mult_fast=1.2,  # 1H ATR×1.2 adaptive pullback zone (0=use pct fallback)
        bias_gate_fast=65.0,      # MTF bias gate — raised 60→65 to cut marginal entries (XAG false-flip fix)
        tp2_r_fast=2.5,           # FAST runner target 2.5R (closes faster than 3R; BULL
                                  # health can still extend toward the 3.0R ladder max)
        cooldown_bars=5,          # whipsaw cooldown: block N×15m after signal (time-based, not call-based)
        # crash-guard
        health_guard_enabled=True,
        health_underwater_frac=0.7,
        health_bias_flip=50.0,    # used in strict mode; fast mode uses bias_gate_fast
        # reversal spike-cut (3m × 4 bars = 12 min window)
        reversal_spike_enabled=True,
        reversal_spike_atr=1.5,   # adverse move must be ≥ N×ATR(3m)
        reversal_spike_bars=4,    # window in 3m bars (4×3m = 12 min)
        # Underwater floor before spike-cut may fire. Guard-sim sweep Jan-May 2026
        # (BTC+XAU, entries from the validated production config):
        #   floor 0.0R: WR 22.3% PnL -$112.91 PF 0.41 — fired 84×, mostly at ~0.1R
        #     (quiet-market ATR(3m) shrinks → 1.5×ATR trips on ordinary noise)
        #   floor 0.5R: PF 1.23 | 0.6R: PF 1.35 | 0.7R: PF 1.46 (+$94.10, fires 4×)
        #   floor 0.8R: PF 1.46 (+$93.89, plateau) | spike-off: PF 1.43 (+$90.04)
        # 0.7R turns spike-cut into a true emergency brake (4 fires in 5 months) that
        # BEATS having no spike-cut — it still catches real crashes (SL count 3→1).
        reversal_spike_uw_frac=0.7,
        # trend-fade cut: early-loss exit when trend dies AND price turns AND losing
        # Backtest Jan-May 2026: Classic +16% ($241→$279), SJ +52% ($143→$217),
        # both with lower MaxDD. Triple gate avoids chopping quiet winners.
        trend_fade_enabled=True,
        trend_fade_uw_frac=0.6,   # min R underwater before the cut can fire
        # reversal-cut: catches a grinding V-reversal that crash-guard (needs HTF
        # flip) and trend-fade (needs ADX falling) both structurally miss because
        # a fresh reversal often makes ADX RISE and HTF bias lags. Experimental;
        # off by default pending backtest.
        reversal_cut_enabled=False,
        reversal_cut_uw_frac=0.7,
        # SJ Hybrid scoring mode (FAST mode only) — ACTIVE by default.
        # Backtest Jan-May 2026: Hybrid=+$103 vs Classic=+$90 combined (BTC+XAU);
        # with Trend-Fade guard, SJ Hybrid reaches +$217 vs Classic +$279 — but SJ
        # has the far better risk-adjusted profile (PnL/DD 1.76, MaxDD -$123).
        # Swaps EMA9/EMA20 → HMA/EMA5>SMA9 + adds Breakout(10) as 5th component.
        # min_score stays 4 (out of 5 components instead of 4 → more flexible).
        sj_scoring=True,
        hma_period=16,   # HMA16 — backtest Jan-May 2026 shows +8% PnL vs HMA20, lower MaxDD
        macd_slope_gate=False,  # gate 4h entries on MACD histogram slope (anti-noise for fast EMAs)
        # Entry-timing relax knobs (default = legacy behaviour, no change):
        breakout_lookback=2,       # 2-bar high/low — 1 bar faster entry vs 3-bar
        adx_rising_or_strong=0,    # 0=off; if >0, ADX gate accepts (rising OR adx>this)
        # HTF anti-false-flip: N consecutive 15m bars of consistent 4H+1H direction required.
        # 2=30min (default), 4=60min. Prevents early-mode single-bar flips (XAG scenario).
        htf_stability_bars=2,
        # MACD histogram gate: rising over N bars (sign-agnostic) → enter early, not after peak.
        # Caught the XAG 19:47 case: MACD declining at entry = trade already past peak.
        macd_hist_rising_gate=True,
        macd_hist_lookback=2,   # bars back for the rising check (1 = softer/sooner, 2 = current)
        # MACD-histogram peak/exhaustion filter: block entries where the histogram
        # has already peaked and is rolling over (buying a weak, exhausting bounce).
        # Opt-in; lagging by nature (peak confirmed slope_bars later). A/B tested.
        macd_peak_filter=False,
        macd_peak_lookback=6,       # bars to look back for a higher histogram peak
        macd_peak_slope_bars=2,     # bars over which the post-peak decline is measured
        # Momentum confirmation source (the laggiest entry gate): "macd" (old),
        # "volmom" (price up + volume>=MA20, reacts 1 bar), "roc3", "volmom_macd".
        # Backtest Jan-May 2026 (BTC+XAG+XAU), train/test split — volmom is the
        # standout on the HOLDOUT (Apr-May, never tuned on):
        #   macd  : TRAIN PF 1.95 → TEST PF 1.33  WR 82.0%  SL 18.0%  (big train→test drop)
        #   volmom: TRAIN PF 1.51 → TEST PF 1.49  WR 83.7%  SL 16.3%  (train≈test = robust)
        # volmom wins the holdout on PF, WR, SL rate AND PnL despite ~15% fewer
        # trades — the volume-participation filter reacts in 1 bar (vs MACD's
        # EMA 12/26/9 smoothing) and cuts low-conviction fake moves, so false
        # signals go DOWN, not up. The train≈test stability is the tell that this
        # is a genuine (non-overfit) edge, unlike macd's regime-dependent 1.95→1.33.
        momentum_gate_mode="volmom",
        # Startup warmup: block signals for N min after bot/strategy restart.
        # Prevents premature entries before the strategy has enough context.
        startup_warmup_min=45,
        # HTF macro/mid direction mode — 'cross' (legacy 20/50), 'slope', or 'early'.
        # 'slope'/'early' flip the 4h+1h gate before the full crossover → earlier entries.
        # 'early' is the validated production default: flips the 4h+1h gate one bar
        # before the full 20/50 cross → fixes late entries. Backtest Jan-May 2026:
        # combined +$488 vs +$447 cross (+9.1% PnL, WR ↑, MaxDD ↓). XAU benefits most.
        htf_macro_mode="early",
        htf_slope_bars=2,          # bars for EMA-fast slope in 'slope'/'early' modes (legacy fallback)
        htf_slope_pct=0.15,        # EMA-fast must move ≥0.15% in one bar for early flip (0=legacy slope_bars)
        htf_sep_guard=False,       # gate early/slope flip on widening EMA gap (anti-whipsaw)
        # 4H Regime Mode: 2/3 score replaces EMA20/50 cross (no EMA50 lag).
        # Components: [1] EMA20 slope (2-bar lookback) + [2] ADX>18 + [3] price side of EMA20.
        # Backtest vs 'early' mode: run /backtest to compare before enabling in production.
        htf_auto_regime=True,        # fast_mode→2/3vote gate, strict→weighted≥65 gate (auto routing)
        htf_regime_4h=False,        # explicit 2/3 vote override (ignored when htf_auto_regime=True)
        htf_regime_4h_mode=None,    # "weighted" → explicit weighted override
        htf_regime_4h_threshold=65, # weighted score floor (backtest sweep: 65 beats 70 on PnL+WR)
        htf_regime_adx_min=18,      # ADX(4H) minimum for 2/3 vote component [2]
        htf_regime_adx_len=14,      # ADX period for regime checks
        # SJ ROC9: adds ROC(9) direction as 6th component (5 → 6).
        # Backtest Jan-May 2026: min5/6 → combined +$447 vs baseline +$401 (+11.5% PnL, same MaxDD).
        sj_roc9=True,
        # SJ Extended scoring: Market Structure + FVG (obv_trend moved to sj_obv)
        sj_extended=False,         # enable 2 extra 15m components; adjust min_score to 6+
        sj_vol_expansion=False,    # DISABLED: redundant with rel_vol (both measure volume elevation)
        # OBV trend: smart money flow direction (replaces vol_ok + vol_expansion duo)
        # OBV measures cumulative buy/sell pressure — far less noisy than raw volume spikes.
        sj_obv=True,               # OBV above/below its EMA20 as SJ scoring component (5 pts)
        # BOS: Break of Structure — close breaks beyond N-bar swing high/low
        sj_bos=True,               # BOS as SJ scoring component (15 pts)
        bos_lookback=2,            # 2×15m = 30 min swing — 1 bar faster than previous 3-bar
        # Confidence Score extras — structural quality filter
        sj_ema_slope=True,         # EMA20 + EMA50 both rising = structural trend (10 pts)
        sj_adx_rising=True,        # ADX rising over 2 bars = trend strengthening (5 pts)
        sj_atr_expansion=True,     # ATR14 > ATR_MA20 = volatility expanding, not quiet (5 pts)
        # Simple Fast Entry: replaces pullback zone / chop / regime / MACD-rising / trigger
        # with just: close vs HMA16 + MACD hist sign + ADX > min (fast mode only).
        # Keeps 4H/1H direction, MTF bias, and score gate intact.
        simple_fast_entry=False,
        # Score-Primary Entry (experimental): ADX-only hard gate, score fully decides
        # direction/momentum/structure. Distinct from simple_fast_entry — see _compute().
        score_primary_entry=False,
        # Pullback timing granularity: compare the LIVE 15m close vs 1H EMA20 instead
        # of the stale last-closed 1H bar. Backtest Jan-May 2026 (BTC+XAG+XAU) proved
        # this WORSE: T=73 PnL=+$46.80 PF=1.22 vs baseline T=117 PnL=+$158.53 PF=1.50.
        # The stale h1_close acts as a useful time-filter (1H bar must CLOSE in the
        # zone) — live 15m is too fast and enters on intra-hour noise. Kept off.
        pullback_live_15m=False,
        # MTF bias source: exclude the noisy 15m self-vote, use 1h+4h only. Backtest
        # Jan-May 2026 (BTC+XAG+XAU): T=114 PnL=+$161.49 PF=1.51 vs baseline T=117
        # PnL=+$158.53 PF=1.50 — better on every metric (15m TF is already filtered
        # by the ADX/score/MACD gates, so its vote just added noise to the HTF gate).
        bias_htf_only=True,
        # ATR compression gate: requires ATR14 to have recently been < ATR50 then start expanding
        # Catches volatility squeeze → expansion setups (disabled by default pending backtest)
        atr_compress_gate=False,
        # Final entry trigger: close must exceed previous bar's high (long) / low (short)
        # Adds precise timing confirmation after all other layers pass
        final_trigger_enabled=True,
        # Trigger mode: "strict" (break prior high/low), "relaxed" (rise over prior
        # close — enters ~1 bar sooner), "off" (no confirmation). Must be in DEFAULTS
        # so it flows into _p. None → derive from final_trigger_enabled (legacy).
        # Backtest Jan-May 2026 (BTC+XAG+XAU) — relaxed DOMINATES on every metric:
        #   strict : T=114 WR=79.8% PnL=+161.49 PF=1.51  (old default)
        #   relaxed: T=139 WR=80.6% PnL=+259.22 PF=1.70  ← faster AND more accurate
        #   off    : T=178 WR=78.1% PnL=+235.77 PF=1.45
        # relaxed enters ~1 bar sooner than strict (the #1 entry bottleneck per the
        # gate-bottleneck diagnostic — 804 sole-blocks) but still filters bars whose
        # momentum is ticking against the trade, which is why it beats off too.
        final_trigger_mode="relaxed",
        pressure_body_ratio=0.60,   # "pressure" mode: min close-in-bar ratio for the expansion bar
        # HTF ADX gate: require 1h AND 4h ADX > threshold alongside 15m ADX
        htf_adx_gate=False,
        htf_adx_len=14,
        htf_adx_min_1h=20,
        htf_adx_min_4h=18,
        # ── V3 Ultra improvements ─────────────────────────────────────────────────
        # Phase 1: Speed & Precision
        # HMA20 pullback anchor (replaces EMA20 in the 1H pullback zone, fast mode).
        # Backtest Jan-May 2026 (BTC+XAG+XAU): HMA anchor T=117 WR=80.3% PnL=+$158.53
        # PF=1.50 vs EMA20 anchor T=120 WR=80.0% PnL=+$149.56 PF=1.45 — HMA responds
        # faster to price so the zone tracks more tightly; strictly better on every
        # metric (fewer trades, higher WR, higher PnL, higher PF), no trade-off found.
        use_hma20_pullback=True,
        sj_zlema=False,             # ZLEMA9 vs ZLEMA20 alignment as SJ scoring component
        sj_price_action=False,      # pin bar / engulfing candle as SJ soft score
        # Phase 2: Accuracy
        chop_filter_enabled=True,   # 1H Chop Index gate: LOW(<threshold)=trending, block choppy
        chop_threshold=50.0,        # chop < threshold = trending market (50=neutral, 38=very trending)
        vol_boost_gate=False,       # require entry bar volume > MA20 × vol_boost_mult
        vol_boost_mult=1.5,         # volume boost multiplier (1.5x = 50% above MA20)
        sj_rsi_div=False,           # hidden RSI divergence as SJ soft score (no lookahead)
        rsi_div_lookback=5,         # lookback bars for divergence comparison
        # Phase 3: Adaptability
        dynamic_sl_enabled=False,   # ATR14/ATR50 ratio → widen SL in high-vol, tighten in chop
        dynamic_sl_high_mult=1.3,   # sl_mult × this when ATR14/ATR50 > 1.3 (volatile)
        dynamic_sl_low_mult=0.8,    # sl_mult × this when ATR14/ATR50 < 0.8 (quiet/choppy)
        # TP1 breakeven buffer: after partial close, set runner SL to entry + N×R instead of exact BE.
        # Backtest Jan-May 2026 E:tp25_be25 winner: PnL +$121 vs +$88 baseline (+38%), PF 1.62 vs 1.45.
        tp1_be_buffer_r=0.25,       # 0.25R buffer above entry for runner stop (0=exact breakeven)
        # ── Market Regime ──────────────────────────────────────────────────────
        # Classifies each bar: STRONG_TREND / TREND / HIGH_VOL / RANGE / LOW_VOL / EXHAUSTION.
        # Blocks RANGE/LOW_VOL/EXHAUSTION entries; widens SL in HIGH_VOL.
        market_regime_enabled=True,
        regime_atr_pct_window=100,  # rolling window for ATR percentile rank
        regime_low_vol_pct=20,      # ATR percentile < 20 → LOW_VOL (skip)
        regime_high_vol_pct=80,     # ATR percentile > 80 → HIGH_VOL (enter, wider SL)
        regime_strong_adx=30,       # ADX threshold for STRONG_TREND
        regime_trend_adx=18,        # ADX threshold for TREND (below = RANGE)
        regime_strong_chop=38,      # Chop(1H) ceiling for STRONG_TREND
        regime_sl_high_vol=1.5,     # SL multiplier in HIGH_VOL (×1.5 wider)
        # Regime-adaptive min_score: relax the bar in STRONG_TREND, tighten in HIGH_VOL.
        # Backtest Jan-May 2026 (BTC+XAG+XAU): T=131 PnL=+202.11 PF=1.57 vs baseline
        # T=139 PnL=+259.22 PF=1.70 — WORSE. The HIGH_VOL tightening (70) removed
        # profitable score-60-69 trades, costing more than the STRONG_TREND relaxation
        # gained. Kept off; 60-69 entries are already good quality.
        regime_adaptive_min_score=False,
        min_score_strong=50.0,      # min_score when regime == STRONG_TREND
        min_score_highvol=70.0,     # min_score when regime == HIGH_VOL
        # Strong-trend breakout entry: waive ONLY the pullback-zone requirement when
        # regime == STRONG_TREND and score ≥ strong_trend_score — catches runs that
        # never pull back to the 1H zone. All other gates still apply.
        # Backtest Jan-May 2026 (BTC+XAG+XAU): T=140 WR=80.7% PnL=+261.24 PF=1.71 vs
        # baseline T=139/80.6%/+259.22/1.70 — better on every metric, no trade-off.
        # Effect is small in this sample but it is the only entry path for trends
        # that run without ever pulling back to the 1H zone (the classic missed-run
        # case); in strongly trending months it matters more. Unlike the failed
        # score_primary_entry (waived every gate everywhere), this waives one gate
        # in the single best regime under a much higher score bar.
        strong_trend_entry=True,
        strong_trend_score=80.0,
        # ── Relative Volume (SJ scoring component) ─────────────────────────────
        # Entry bar volume relative to MA20(volume). Filters fake/low-interest breakouts.
        sj_rel_vol=True,            # rel_vol: volume/MA20 ≥1.2x — the single volume quality gate
        rel_vol_min=1.2,            # entry bar must have ≥1.2× average volume
        # HTF Momentum Score — 8-component quality gate (replaces binary crossover when enabled)
        htf_mom_score=False,       # off by default; enable to replace binary macro/mid gates
        min_htf_score=6,           # require 6/8 for entry (5 = moderate, 6 = strict)
        htf_sep_thresh_4h=0.002,   # 4h EMA separation min: 0.20% of price
        htf_sep_thresh_1h=0.001,   # 1h EMA separation min: 0.10% of price
        htf_rsi_bull_lo=45.0,      # 4h/1h RSI bull zone lower bound (long entries)
        htf_rsi_bull_hi=75.0,      # 4h/1h RSI bull zone upper bound
        htf_rsi_bear_lo=25.0,      # 4h/1h RSI bear zone lower bound (short entries)
        htf_rsi_bear_hi=55.0,      # 4h/1h RSI bear zone upper bound
        # ema5_sma9: OFF by default — overlaps with hma_bull (both are fast-vs-slow
        # direction checks), adds no distinct signal, just noise on the score total.
        sj_ema5_sma9=False,
        # macd_hist: MACD line vs signal as an explicit score component (5 pts) —
        # fills the 5 pts freed by disabling ema5_sma9, keeping max score at 100.
        # Mirrors the same MACD hist sign already used by simple_fast_entry.
        sj_macd_hist=True,
    )

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self._p = {**self.DEFAULTS, **{k: self.params.get(k, v) for k, v in self.DEFAULTS.items()}}
        sj   = bool(self._p.get("sj_scoring",        False))
        ext  = bool(self._p.get("sj_extended",       False))
        r9   = bool(self._p.get("sj_roc9",           False))
        ve   = bool(self._p.get("sj_vol_expansion",  False))
        bos  = bool(self._p.get("sj_bos",            False))
        zl   = bool(self._p.get("sj_zlema",          False))
        pa   = bool(self._p.get("sj_price_action",   False))
        rdiv = bool(self._p.get("sj_rsi_div",        False))
        rv   = bool(self._p.get("sj_rel_vol",        False))
        obv  = bool(self._p.get("sj_obv",            False))
        esl  = bool(self._p.get("sj_ema_slope",      False))
        adr  = bool(self._p.get("sj_adx_rising",     False))
        atrx = bool(self._p.get("sj_atr_expansion",  False))
        e59  = bool(self._p.get("sj_ema5_sma9",      True))
        mh   = bool(self._p.get("sj_macd_hist",      False))
        self._p["indicators"] = self.params.get("indicators",
            build_indicator_registry(sj_scoring=sj, extended=ext, roc9=r9, vol_expansion=ve,
                                     bos=bos, zlema=zl, price_action=pa, rsi_div=rdiv,
                                     rel_vol=rv, obv=obv, ema_slope=esl,
                                     adx_rising_score=adr, atr_expansion=atrx,
                                     ema5_sma9=e59, macd_hist_score=mh))
        self._min_primary = self.params.get("min_primary", 100)
        self._min_1h      = self.params.get("min_1h",       60)
        self._min_4h      = self.params.get("min_4h",       55)
        self._cooldown_until = 0.0  # unix timestamp when cooldown expires (time-based, not call-based)
        self._start_time     = time.time()  # for startup warmup check

    def arm_cooldown(self) -> None:
        """Start the whipsaw cooldown. Called by the bot only after a confirmed
        fill, so a rejected order never blocks future signals."""
        if self._p.get("fast_mode") and self._p.get("cooldown_bars", 0) > 0:
            self._cooldown_until = time.time() + self._p["cooldown_bars"] * 15 * 60

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
                         candles_5m: list, candles_1h: list, candles_4h: list,
                         candles_3m: list = None, candles_15m: list = None) -> tuple[str, str]:
        """
        Guard stack (checked in order, first CLOSE wins):
          1. Spike-Cut    — fast adverse ≥1.5×ATR(3m) in 12 min (pre-TP1 only)
          2. Trend-Fade   — ≥0.6R underwater + ADX(15m) falling + lost EMA20(5m) (pre-TP1)
          3. Reversal-Cut — ≥0.7R underwater + EMA9(5m) accelerating against (pre-TP1)
          4. Crash-Guard  — 0.7R underwater + momentum fully reversed
        Returns ('CLOSE'|'HOLD', reason).
        """
        if not self._p["health_guard_enabled"]:
            return ("HOLD", "health guard disabled")
        if not candles_5m or not candles_1h or not candles_4h:
            return ("HOLD", "no candle data")

        # ── Layer 1: Spike-Cut (3m, pre-TP1 only) ─────────────────────────────
        if (self._p.get("reversal_spike_enabled") and not tp1_hit
                and candles_3m and one_r > 0):
            df3m = self._to_df(candles_3m)
            sc_action, sc_reason = check_spike_cut(
                side, entry, one_r, df3m,
                spike_atr_mult=self._p["reversal_spike_atr"],
                spike_bars=self._p["reversal_spike_bars"],
                uw_frac=self._p["reversal_spike_uw_frac"],
            )
            if sc_action == "CLOSE":
                return ("CLOSE", sc_reason)

        df5  = self._to_df(candles_5m)
        df1h = self._to_df(candles_1h)
        df4h = self._to_df(candles_4h)

        # ── Layer 2: Trend-Fade Cut (pre-TP1 — after TP1 the BE stop guards) ───
        # Cuts a dying+turning loser at ~−0.6R instead of the full −1R stop.
        if (self._p.get("trend_fade_enabled") and not tp1_hit
                and candles_15m and one_r > 0):
            df15 = self._to_df(candles_15m)
            tf_action, tf_reason = check_trend_fade(
                side, entry, one_r, df15, df5,
                uw_frac=self._p["trend_fade_uw_frac"],
            )
            if tf_action == "CLOSE":
                return ("CLOSE", tf_reason)

        # ── Layer 3: Reversal-Cut (pre-TP1) ────────────────────────────────────
        # Catches a grinding V-reversal that trend-fade (needs ADX falling) and
        # crash-guard (needs HTF bias to flip) both structurally miss.
        if (self._p.get("reversal_cut_enabled") and not tp1_hit and one_r > 0):
            rc_action, rc_reason = check_reversal_cut(
                side, entry, one_r, df5,
                uw_frac=self._p["reversal_cut_uw_frac"],
            )
            if rc_action == "CLOSE":
                return ("CLOSE", rc_reason)

        # ── Layer 4: Crash-Guard (0.7R + full reversal) ────────────────────────
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

        # ── Startup warmup: block signals for N min after strategy restart ───────
        _warmup_min = float(self._p.get("startup_warmup_min", 0))
        if _warmup_min > 0:
            _elapsed = time.time() - self._start_time
            if _elapsed < _warmup_min * 60:
                _rem = int((_warmup_min * 60 - _elapsed) / 60)
                return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                              f"[TCImproved] startup warmup: {_rem}min left")

        # ── Whipsaw cooldown (fast mode only, time-based) ─────────────────────
        if self._p.get("fast_mode") and time.time() < self._cooldown_until:
            remaining_min = int((self._cooldown_until - time.time()) / 60)
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[TCImproved] whipsaw cooldown: {remaining_min}min left")

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
            gate   = self._p["bias_gate_fast"] if self._p.get("fast_mode") else self._p["bias_gate"]
            _y = lambda k: "✓" if last.get(k) else "✗"
            long_layers  = f"4H{_y('d_macro_up')} 1H{_y('d_mid_up')} pull{_y('d_pull_l')} bias{_y('d_bias_l')} adx{_y('d_adx_ok')}"
            short_layers = f"4H{_y('d_macro_dn')} 1H{_y('d_mid_dn')} pull{_y('d_pull_s')} bias{_y('d_bias_s')} adx{_y('d_adx_ok')}"
            adx_max  = self._p.get("adx_max_fast", 50) if self._p.get("fast_mode") else 999
            adx_min  = self._p.get("adx_min_fast", 15) if self._p.get("fast_mode") else self._p.get("adx_min", 30)
            sj_tag   = "[SJ]" if self._p.get("sj_scoring") else ""
            min_sc   = self._p.get("min_score", 60)
            # Count actually-enabled scoring components (5 SJ base + ROC9 + 3 extended)
            _reg     = self._p.get("indicators", {})
            n_comps  = sum(1 for v in _reg.values() if v.get("enabled", True)) or \
                       (5 if self._p.get("sj_scoring") else 4)
            _yi = lambda k, df=True: "✓" if last.get(k, df) else "✗"
            # Inner gates (regime/chop symmetric; macd/trigger directional)
            inner_l = f"reg{_yi('d_regime_ok')} chp{_yi('d_chop_ok')} mac{_yi('d_macd_l')} trg{_yi('d_trig_l')}"
            inner_s = f"reg{_yi('d_regime_ok')} chp{_yi('d_chop_ok')} mac{_yi('d_macd_s')} trg{_yi('d_trig_s')}"
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[TCImproved{sj_tag}] bias={bias_v:.0f}(gate±{gate:.0f}) ADX={adx_v:.0f}({adx_min:.0f}-{adx_max:.0f}) "
                          f"score={sc_l:.0f}L/{sc_s:.0f}S(need≥{min_sc:.0f}/{n_comps}) "
                          f"| L:{long_layers} {inner_l} | S:{short_layers} {inner_s}")

        dist = float(last["dist"])
        if dist <= 0 or np.isnan(dist):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[TCImproved] invalid dist")

        # NOTE: whipsaw cooldown is armed by the bot via arm_cooldown() AFTER a
        # confirmed fill — never here, so a rejected order (e.g. insufficient
        # margin) cannot strand the symbol in cooldown with no open position.

        p = self._p

        # FAST v2 targets tp2_r_fast (default 2.5R, health ladder can extend further); STRICT uses tp2_r (2.5R).
        tp2_r = p.get("tp2_r_fast", p["tp2_r"]) if p.get("fast_mode") else p["tp2_r"]

        def _meta(side: str) -> dict:
            sl  = current_price - dist if side == "long" else current_price + dist
            tp1 = current_price + p["tp1_r"] * dist if side == "long" else current_price - p["tp1_r"] * dist
            tp2 = current_price + tp2_r * dist if side == "long" else current_price - tp2_r * dist
            sc  = float(last.get("score_long", 0) if side == "long" else last.get("score_short", 0))
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
                "sj_score":    sc,
                "confidence_score": sc,
                "confidence_level": "strong" if sc >= 90 else "normal" if sc >= 70 else "small",
                "mtf_bias":    float(last.get("comp_pct", 0)),
                "regime_score_w": float(last.get("h4_regime_score_w", 0)),
                # Lesson-tracker context: which entry path fired and market regime at entry
                "entry_path":  ("strong_trend"
                                if bool(last.get("d_stw_l" if side == "long" else "d_stw_s", False))
                                else "pullback"),
                "regime":      str(last.get("regime", "")),
                "adx15":       float(last.get("adx15", 0) or 0),
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
