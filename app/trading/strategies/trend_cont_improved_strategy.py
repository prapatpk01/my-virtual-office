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
                             price_action: bool = False, rsi_div: bool = False):
    if sj_scoring:
        # SJ Hybrid: faster/smarter components from SJ Fast Entry research
        reg = {
            "hma_bull": dict(
                enabled=True, weight=1.0,
                long =lambda c: c["close"] > c["hma"],
                short=lambda c: c["close"] < c["hma"],
                desc="15m close vs HMA (Hull MA — period set by hma_period param)",
            ),
            "ema5_sma9": dict(
                enabled=True, weight=1.0,
                long =lambda c: c["ema5"] > c["sma9"],
                short=lambda c: c["ema5"] < c["sma9"],
                desc="EMA5 vs SMA9 alignment (more sensitive than EMA9 vs EMA20)",
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
            "breakout_hh10": dict(
                enabled=True, weight=1.0,
                long =lambda c: c["close"] > c["hh10"],
                short=lambda c: c["close"] < c["ll10"],
                desc="15m close breaks above/below N-bar high/low (breakout_lookback param)",
            ),
        }
        if roc9:
            reg["roc9"] = dict(
                enabled=True, weight=1.0,
                long =lambda c: c["roc9"] > 0,
                short=lambda c: c["roc9"] < 0,
                desc="ROC(9) positive/negative — confirms price momentum direction",
            )
        if vol_expansion:
            reg["vol_expansion"] = dict(
                enabled=True, weight=1.0,
                long =lambda c: c["vol_expansion_ok"],
                short=lambda c: c["vol_expansion_ok"],
                desc="Volume > MA20 × 1.15 — breakout volume expansion bonus",
            )
        if extended:
            # SJ Extended (+3 components → 8 total): OBV pressure, market structure,
            # Fair Value Gap imbalance.  Suggest min_score=5 or 6 out of 8.
            reg.update({
                "obv_trend": dict(
                    enabled=True, weight=1.0,
                    long =lambda c: c["obv"] > c["obv_ema20"],
                    short=lambda c: c["obv"] < c["obv_ema20"],
                    desc="OBV above/below its EMA20 (dominant volume pressure)",
                ),
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
                enabled=True, weight=1.0,
                long =lambda c: c["bos_bull"],
                short=lambda c: c["bos_bear"],
                desc="Break of Structure: 15m close exceeds N-bar swing high/low (bos_lookback param)",
            )
        if zlema:
            reg["zlema_dir"] = dict(
                enabled=True, weight=1.0,
                long =lambda c: c["zlema9"] > c["zlema20"],
                short=lambda c: c["zlema9"] < c["zlema20"],
                desc="ZLEMA9 vs ZLEMA20 alignment (zero-lag EMA, faster than EMA5>SMA9)",
            )
        if price_action:
            reg["price_action"] = dict(
                enabled=True, weight=1.0,
                long =lambda c: c["pa_bull"],
                short=lambda c: c["pa_bear"],
                desc="Pin bar or engulfing candle confirmation on 15m",
            )
        if rsi_div:
            reg["rsi_div"] = dict(
                enabled=True, weight=1.0,
                long =lambda c: c["hidden_bull_div"],
                short=lambda c: c["hidden_bear_div"],
                desc="Hidden RSI divergence: price higher/lower but RSI diverges (no lookahead)",
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

    # ── Layer 3: Pullback zone (mode-specific) ────────────────────────────────
    if fast_mode:
        # Fast Mode v2: 1H EMA20 zone (or HMA20 if use_hma20_pullback=True)
        if p.get("use_hma20_pullback", False):
            # Compute on actual 1H bars then reindex — avoids ffill HMA distortion
            ema20_1h = _hma(df1h["close"], 20).reindex(out.index, method="ffill")
        else:
            ema20_1h = ef1
        pullback_atr_mult = float(p.get("pullback_atr_mult_fast", 1.2))
        if pullback_atr_mult > 0:
            # ATR-based: adaptive to volatility (BTC 1H ATR ~$600-1k; 1.2× ~ $720-1.2k)
            h1_atr = _atr(df1h, 14).shift(1)  # shift 1 bar: no-lookahead
            atr_1h = h1_atr.reindex(out.index, method="ffill").bfill()
            near_ema = (out["h1_close"] - ema20_1h).abs() <= atr_1h * pullback_atr_mult
        else:
            pullback_pct = p["pullback_pct_fast"]
            near_ema = (out["h1_close"] - ema20_1h).abs() / ema20_1h <= pullback_pct
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

    # ── Layer 6: Micro score indicators ──────────────────────────────────────
    ema9  = _ema(out["close"], p["ema_micro"])
    ema20 = _ema(out["close"], p["ema_fast"])
    rsi15 = _rsi(out["close"], p["rsi_period"])
    volma = _sma(out["volume"], p["vol_period"])
    vol_ok = (volma > 0) & (out["volume"] >= volma * p["vol_mult"])
    vol_expansion_ok = (volma > 0) & (out["volume"] >= volma * 1.15)

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

    ctx = dict(
        close=out["close"], open=out["open"], ema9=ema9, ema20=ema20, rsi15=rsi15, vol_ok=vol_ok,
        macd=macd_line, macd_signal=macd_sig,
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

    # ── ATR compression gate (optional, separate from SJ scoring) ────────────
    if p.get("atr_compress_gate", False):
        long_gates  = adx_ok & atr_compress_ok
        short_gates = adx_ok & atr_compress_ok
    else:
        long_gates  = adx_ok
        short_gates = adx_ok

    # ── Chop Index gate: block entries in ranging/sideways 1H market ─────────
    if p.get("chop_filter_enabled", False):
        _chop_thresh = float(p.get("chop_threshold", 50.0))
        _chop_1h = _chop_index(df1h, 14).reindex(out.index, method="ffill").fillna(50.0)
        out["chop_1h"] = _chop_1h
        _chop_ok = _chop_1h < _chop_thresh   # LOW = trending; HIGH = choppy
        long_gates  = long_gates  & _chop_ok
        short_gates = short_gates & _chop_ok

    # ── Volume boost gate: entry bar must have elevated volume ────────────────
    if p.get("vol_boost_gate", False):
        _vb_mult = float(p.get("vol_boost_mult", 1.5))
        _vol_boost_ok = (volma > 0) & (out["volume"] >= volma * _vb_mult)
        long_gates  = long_gates  & _vol_boost_ok
        short_gates = short_gates & _vol_boost_ok

    # ── Final entry trigger: close must break above previous bar high/low ─────
    if p.get("final_trigger_enabled", False):
        trigger_long  = out["close"] > out["high"].shift(1)
        trigger_short = out["close"] < out["low"].shift(1)
    else:
        trigger_long  = pd.Series(True, index=out.index)
        trigger_short = pd.Series(True, index=out.index)

    out["final_buy"]  = (macro_up & mid_up  & at_pull_long  & long_bias_ok  &
                          (score_long  >= p["min_score"]) & long_gates  & trigger_long).fillna(False)
    out["final_sell"] = (macro_dn  & mid_dn  & at_pull_short & short_bias_ok &
                          (score_short >= p["min_score"]) & short_gates & trigger_short).fillna(False)

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

    # ── Dynamic SL: ATR14/ATR50 ratio adjusts sl_mult to market volatility ──────
    if p.get("dynamic_sl_enabled", False):
        _atr50_sl   = _rma(_true_range(out), 50)
        _atr_ratio  = (out["atr"] / _atr50_sl.replace(0, np.nan)).fillna(1.0)
        _dyn_hi     = float(p.get("dynamic_sl_high_mult", 1.3))
        _dyn_lo     = float(p.get("dynamic_sl_low_mult",  0.8))
        _sl_mult_v  = pd.Series(
            np.where(_atr_ratio > 1.3, p["sl_mult"] * _dyn_hi,
            np.where(_atr_ratio < 0.8, p["sl_mult"] * _dyn_lo,
                     p["sl_mult"])),
            index=out.index)
        dist = (out["atr"] * _sl_mult_v).clip(
            lower=out["close"] * p["sl_min_pct"],
            upper=out["close"] * p["sl_max_pct"],
        )
    else:
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


# ── REVERSAL SPIKE-CUT ────────────────────────────────────────────────────────

def check_spike_cut(side: str, entry: float, one_r: float,
                    df3m: pd.DataFrame,
                    spike_atr_mult: float = 1.5,
                    spike_bars: int = 4) -> tuple[str, str]:
    """
    Fast adverse-move detector using 3m candles.
    Fires when price moves ≥ spike_atr_mult×ATR(3m) against the position
    within spike_bars×3m (default 4×3m = 12 min). No minimum loss required.
    """
    min_bars = spike_bars + 15          # need enough history for ATR(14)
    if df3m is None or len(df3m) < min_bars:
        return ("HOLD", "insufficient 3m data for spike-cut")

    atr3 = float(_atr(df3m, 14).iloc[-1])
    if atr3 <= 0:
        return ("HOLD", "invalid ATR(3m) — spike-cut skipped")

    px_now  = float(df3m["close"].iloc[-1])
    px_prev = float(df3m["close"].iloc[-(spike_bars + 1)])

    # Adverse move: down for long, up for short
    adverse_move = (px_prev - px_now) if side == "long" else (px_now - px_prev)
    move_r = adverse_move / atr3

    if move_r >= spike_atr_mult:
        uw = ((entry - px_now) / one_r if side == "long"
              else (px_now - entry) / one_r)
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
        min_score=5.0,   # 5/8 with sj_bos=True — backtest: +$103.94 vs 6/8=$62.64 (5/8 wins by $41)
        # Fast mode v2
        fast_mode=False,
        adx_min_fast=15,          # ADX lower bound — trend must be active. Lowered 18→15
                                  # (keep adx_rising_fast=True) to enter 1-2 bars earlier on
                                  # a fresh trend; the "rising" gate still blocks dead chop.
        adx_max_fast=45,          # ADX upper bound — caps overheated entries.
        adx_rising_fast=True,     # also require ADX[0] > ADX[1]
        pullback_pct_fast=0.025,  # fallback pct zone (used when pullback_atr_mult_fast=0)
        pullback_atr_mult_fast=1.2,  # 1H ATR×1.2 adaptive pullback zone (0=use pct fallback)
        bias_gate_fast=60.0,      # MTF bias gate — 60 is live-calibrated (grid winner was 70 but too strict for live market phases)
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
        # trend-fade cut: early-loss exit when trend dies AND price turns AND losing
        # Backtest Jan-May 2026: Classic +16% ($241→$279), SJ +52% ($143→$217),
        # both with lower MaxDD. Triple gate avoids chopping quiet winners.
        trend_fade_enabled=True,
        trend_fade_uw_frac=0.6,   # min R underwater before the cut can fire
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
        breakout_lookback=7,       # 7-bar high/low (was 10); fires sooner while filtering noise
        adx_rising_or_strong=0,    # 0=off; if >0, ADX gate accepts (rising OR adx>this)
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
        # SJ Extended scoring: adds OBV trend + Market Structure + FVG (5 → 8 components)
        sj_extended=False,         # enable 3 extra 15m components (total 9); adjust min_score to 5-6
        sj_vol_expansion=True,     # add vol>MA20×1.15 as 7th SJ component (breakout volume bonus)
        # BOS: Break of Structure — close breaks beyond N-bar swing high/low
        # Soft score: adds +1 point; min_score=5 still gates entry, BOS just helps push borderline
        # setups over the threshold without being required.
        sj_bos=True,               # BOS as soft score component (8th component, min_score=6/8)
        bos_lookback=5,            # 5×15m = 75 min swing lookback (try 3-8)
        # ATR compression gate: requires ATR14 to have recently been < ATR50 then start expanding
        # Catches volatility squeeze → expansion setups (disabled by default pending backtest)
        atr_compress_gate=False,
        # Final entry trigger: close must exceed previous bar's high (long) / low (short)
        # Adds precise timing confirmation after all other layers pass
        final_trigger_enabled=True,
        # HTF ADX gate: require 1h AND 4h ADX > threshold alongside 15m ADX
        htf_adx_gate=False,
        htf_adx_len=14,
        htf_adx_min_1h=20,
        htf_adx_min_4h=18,
        # ── V3 Ultra improvements ─────────────────────────────────────────────────
        # Phase 1: Speed & Precision
        use_hma20_pullback=False,   # replace EMA20→HMA20 in 1H pullback zone (actual 1H bars)
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
        # HTF Momentum Score — 8-component quality gate (replaces binary crossover when enabled)
        htf_mom_score=False,       # off by default; enable to replace binary macro/mid gates
        min_htf_score=6,           # require 6/8 for entry (5 = moderate, 6 = strict)
        htf_sep_thresh_4h=0.002,   # 4h EMA separation min: 0.20% of price
        htf_sep_thresh_1h=0.001,   # 1h EMA separation min: 0.10% of price
        htf_rsi_bull_lo=45.0,      # 4h/1h RSI bull zone lower bound (long entries)
        htf_rsi_bull_hi=75.0,      # 4h/1h RSI bull zone upper bound
        htf_rsi_bear_lo=25.0,      # 4h/1h RSI bear zone lower bound (short entries)
        htf_rsi_bear_hi=55.0,      # 4h/1h RSI bear zone upper bound
    )

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self._p = {**self.DEFAULTS, **{k: self.params.get(k, v) for k, v in self.DEFAULTS.items()}}
        sj  = bool(self._p.get("sj_scoring",      False))
        ext = bool(self._p.get("sj_extended",     False))
        r9  = bool(self._p.get("sj_roc9",         False))
        ve  = bool(self._p.get("sj_vol_expansion", False))
        bos  = bool(self._p.get("sj_bos",          False))
        zl   = bool(self._p.get("sj_zlema",        False))
        pa   = bool(self._p.get("sj_price_action", False))
        rdiv = bool(self._p.get("sj_rsi_div",      False))
        self._p["indicators"] = self.params.get("indicators",
            build_indicator_registry(sj_scoring=sj, extended=ext, roc9=r9, vol_expansion=ve,
                                     bos=bos, zlema=zl, price_action=pa, rsi_div=rdiv))
        self._min_primary = self.params.get("min_primary", 100)
        self._min_1h      = self.params.get("min_1h",       60)
        self._min_4h      = self.params.get("min_4h",       55)
        self._cooldown_until = 0.0  # unix timestamp when cooldown expires (time-based, not call-based)

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
          1. Spike-Cut   — fast adverse ≥1.5×ATR(3m) in 12 min (pre-TP1 only)
          2. Trend-Fade  — ≥0.6R underwater + ADX(15m) falling + lost EMA20(5m) (pre-TP1)
          3. Crash-Guard — 0.7R underwater + momentum fully reversed
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

        # ── Layer 3: Crash-Guard (0.7R + full reversal) ────────────────────────
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
            adx_min  = self._p.get("adx_min_fast", 18) if self._p.get("fast_mode") else self._p.get("adx_min", 30)
            sj_tag   = "[SJ]" if self._p.get("sj_scoring") else ""
            min_sc   = self._p.get("min_score", 4)
            # Count actually-enabled scoring components (5 SJ base + ROC9 + 3 extended)
            _reg     = self._p.get("indicators", {})
            n_comps  = sum(1 for v in _reg.values() if v.get("enabled", True)) or \
                       (5 if self._p.get("sj_scoring") else 4)
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[TCImproved{sj_tag}] bias={bias_v:.0f}(gate±{gate:.0f}) ADX={adx_v:.0f}({adx_min:.0f}-{adx_max:.0f}) "
                          f"score={sc_l:.0f}L/{sc_s:.0f}S(need≥{min_sc:.0f}/{n_comps}) "
                          f"| L:{long_layers} | S:{short_layers}")

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
                "mtf_bias":    float(last.get("comp_pct", 0)),
                "regime_score_w": float(last.get("h4_regime_score_w", 0)),
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
