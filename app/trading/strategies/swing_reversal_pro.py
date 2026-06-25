"""
SWING REVERSAL PRO V1
Reversal strategy for BTC/USDT & XAU/USDT futures (Hedge Mode, Isolated Margin).

Timeframes:
  Entry     = 15m   (mtf["15m"] or primary candles)
  Context   = 1H    (mtf["1h"])
  Structure = 4H    (mtf["4h"])

Two instances per symbol — one Long, one Short:
  SwingReversalPro(sym, {"direction": "long"})  → name SwingReversalPro_L
  SwingReversalPro(sym, {"direction": "short"}) → name SwingReversalPro_S

Signal metadata:
  position_side = "LONG" | "SHORT"
  action        = "open"  | "close"
  stop_loss, take_profit, tp1, r_dist

Entry layers:
  L1  Reversal Score  ≥ 5/7   (RSI / Divergence / HMA / ADX-rollover /
                                 Volume / Liquidity-sweep / Extension)
  L2  Context filter  ≥ 5/6   (4H ADX / 1H RSI / Volume / S-R zone /
                                 MTF Bias / EMA slope)
  L3  Price-action trigger ≥ 1 (CHOCH / BOS / Hammer / Engulfing /
                                  Double-bottom/top / V-reversal)

Risk:
  SL = wider of (pattern extreme | 1.0×ATR14)
  TP1 = 0.8R  →  close 70%, move SL → breakeven
  TP2 = 1.5R  →  close remaining

Early exit:
  Long : Bearish CHOCH | RSI>75 | new bearish divergence | health<25
  Short: Bullish CHOCH | RSI<25 | new bullish divergence | health<25
"""
import math
from .base import BaseStrategy, Signal, SignalType

_BUY  = SignalType.BUY
_SELL = SignalType.SELL
_HOLD = SignalType.HOLD

_WARMUP_15M = 55   # bars needed before analysis begins
_WARMUP_1H  = 30
_WARMUP_4H  = 20


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rsi_divergence(closes: list, rsi: list, lookback: int = 16) -> str | None:
    """Return 'bullish', 'bearish', or None."""
    if len(closes) < lookback + 2:
        return None
    r = closes[-lookback:]
    rv = rsi[-lookback:]

    # Swing lows (for bullish divergence)
    lows_idx = [i for i in range(1, len(r) - 1)
                if r[i] < r[i - 1] and r[i] < r[i + 1]]
    if len(lows_idx) >= 2:
        i1, i2 = lows_idx[-2], lows_idx[-1]
        if (not math.isnan(rv[i1]) and not math.isnan(rv[i2])
                and r[i2] < r[i1] and rv[i2] > rv[i1]):
            return "bullish"

    # Swing highs (for bearish divergence)
    highs_idx = [i for i in range(1, len(r) - 1)
                 if r[i] > r[i - 1] and r[i] > r[i + 1]]
    if len(highs_idx) >= 2:
        i1, i2 = highs_idx[-2], highs_idx[-1]
        if (not math.isnan(rv[i1]) and not math.isnan(rv[i2])
                and r[i2] > r[i1] and rv[i2] < rv[i1]):
            return "bearish"
    return None


def _hma_slope(hma_arr: list, n: int, bars: int = 3) -> float:
    """Normalised slope of HMA over last `bars` bars (positive = rising)."""
    if n < bars or math.isnan(float(hma_arr[n])) or math.isnan(float(hma_arr[n - bars])):
        return 0.0
    return (float(hma_arr[n]) - float(hma_arr[n - bars])) / bars


def _adx_rollover(adx_arr: list, n: int, lookback: int = 3) -> bool:
    """True when ADX has peaked and is now declining (rollover from top)."""
    if n < lookback + 1:
        return False
    vals = [float(adx_arr[n - i]) for i in range(lookback + 1)]
    if any(math.isnan(v) for v in vals):
        return False
    peak = max(vals[1:])        # max of the look-back window
    return vals[0] < peak and vals[0] < vals[1]


def _liquidity_sweep_low(highs: list, lows: list, closes: list, n: int,
                          lookback: int = 10) -> bool:
    """Bar swept below previous swing low then closed back above it."""
    if n < lookback + 1:
        return False
    swing_low = min(lows[n - lookback: n])
    return lows[n] < swing_low and closes[n] > swing_low


def _liquidity_sweep_high(highs: list, lows: list, closes: list, n: int,
                           lookback: int = 10) -> bool:
    """Bar swept above previous swing high then closed back below it."""
    if n < lookback + 1:
        return False
    swing_high = max(highs[n - lookback: n])
    return highs[n] > swing_high and closes[n] < swing_high


def _hammer(o: float, h: float, l: float, c: float, atr: float) -> bool:
    body = max(abs(c - o), atr * 0.05)
    lower = min(o, c) - l
    upper = h - max(o, c)
    return lower >= 2.0 * body and upper <= body and c >= o


def _shooting_star(o: float, h: float, l: float, c: float, atr: float) -> bool:
    body = max(abs(c - o), atr * 0.05)
    upper = h - max(o, c)
    lower = min(o, c) - l
    return upper >= 2.0 * body and lower <= body and c <= o


def _bull_engulf(opens: list, closes: list, n: int) -> bool:
    if n < 1:
        return False
    p_bear = closes[n - 1] < opens[n - 1]
    c_bull = closes[n] > opens[n]
    engulf = closes[n] > opens[n - 1] and opens[n] < closes[n - 1]
    return p_bear and c_bull and engulf


def _bear_engulf(opens: list, closes: list, n: int) -> bool:
    if n < 1:
        return False
    p_bull = closes[n - 1] > opens[n - 1]
    c_bear = closes[n] < opens[n]
    engulf = closes[n] < opens[n - 1] and opens[n] > closes[n - 1]
    return p_bull and c_bear and engulf


def _double_bottom(lows: list, closes: list, n: int,
                   lookback: int = 20, tol: float = 0.003) -> bool:
    if n < lookback:
        return False
    w = lows[n - lookback: n + 1]
    order = sorted(range(len(w)), key=lambda i: w[i])
    i1, i2 = sorted(order[:2])
    if abs(w[i1] - w[i2]) / max(w[i1], 1e-10) > tol:
        return False
    return closes[n] > (w[i1] + w[i2]) / 2 and i2 == len(w) - 1


def _double_top(highs: list, closes: list, n: int,
                lookback: int = 20, tol: float = 0.003) -> bool:
    if n < lookback:
        return False
    w = highs[n - lookback: n + 1]
    order = sorted(range(len(w)), key=lambda i: -w[i])
    i1, i2 = sorted(order[:2])
    if abs(w[i1] - w[i2]) / max(w[i1], 1e-10) > tol:
        return False
    return closes[n] < (w[i1] + w[i2]) / 2 and i2 == len(w) - 1


def _choch_bull(highs: list, closes: list, n: int, lookback: int = 10) -> bool:
    if n < lookback + 1:
        return False
    swing_high = max(highs[n - lookback: n])
    return closes[n] > swing_high


def _choch_bear(lows: list, closes: list, n: int, lookback: int = 10) -> bool:
    if n < lookback + 1:
        return False
    swing_low = min(lows[n - lookback: n])
    return closes[n] < swing_low


def _bos_bull(highs: list, closes: list, n: int, lookback: int = 5) -> bool:
    if n < lookback + 1:
        return False
    return closes[n] > max(highs[n - lookback: n])


def _bos_bear(lows: list, closes: list, n: int, lookback: int = 5) -> bool:
    if n < lookback + 1:
        return False
    return closes[n] < min(lows[n - lookback: n])


def _v_reversal_long(closes: list, lows: list, n: int, atr: float,
                     lookback: int = 6) -> bool:
    """Sharp drop followed by sharp recovery within lookback bars."""
    if n < lookback * 2:
        return False
    seg = lows[n - lookback: n + 1]
    min_i = seg.index(min(seg))
    if min_i == 0 or min_i == len(seg) - 1:
        return False
    drop     = closes[n - lookback] - seg[min_i]
    recovery = closes[n] - seg[min_i]
    return drop >= atr * 0.8 and recovery >= drop * 0.6


def _v_reversal_short(closes: list, highs: list, n: int, atr: float,
                      lookback: int = 6) -> bool:
    if n < lookback * 2:
        return False
    seg = highs[n - lookback: n + 1]
    max_i = seg.index(max(seg))
    if max_i == 0 or max_i == len(seg) - 1:
        return False
    surge    = seg[max_i] - closes[n - lookback]
    reversal = seg[max_i] - closes[n]
    return surge >= atr * 0.8 and reversal >= surge * 0.6


def _near_support(price: float, lows_4h: list, n4h: int,
                  atr4h: float, lookback: int = 20) -> bool:
    if n4h < lookback:
        return False
    levels = sorted(lows_4h[n4h - lookback: n4h + 1])[:5]
    return any(abs(price - lvl) <= 1.5 * atr4h for lvl in levels)


def _near_resistance(price: float, highs_4h: list, n4h: int,
                     atr4h: float, lookback: int = 20) -> bool:
    if n4h < lookback:
        return False
    levels = sorted(highs_4h[n4h - lookback: n4h + 1])[-5:]
    return any(abs(price - lvl) <= 1.5 * atr4h for lvl in levels)


def _ema20_slope(ema20_arr: list, n: int, bars: int = 4) -> float:
    if n < bars:
        return 0.0
    v0, vb = float(ema20_arr[n]), float(ema20_arr[n - bars])
    return 0.0 if math.isnan(v0) or math.isnan(vb) else (v0 - vb) / bars


def _vol_not_declining(volma: list, n: int, bars: int = 3) -> bool:
    if n < bars:
        return True
    v0, vb = float(volma[n]), float(volma[n - bars])
    if math.isnan(v0) or math.isnan(vb) or vb <= 0:
        return True
    return v0 >= vb * 0.85   # allow up to -15% drift

def _candle_pressure(op: list, cl: list, hi: list, lo: list,
                     n: int, atr14: float, direction: str,
                     bars: int = 2) -> str | None:
    """
    Detect gradual momentum build — N consecutive opposing closes.
    Fires after 2 bars (30 min) to catch 3-4 bar pressure before SL is hit.

    Conditions (both required):
      1) Last `bars` closes all oppose direction (each bar closes against position)
      2) Cumulative close-to-close move ≥ 0.5×ATR14  (not just noise)
      3) Each individual bar's close-to-close move ≥ 0.15×ATR (consistent, not one big bar)
    """
    if n < bars + 1 or atr14 <= 0:
        return None
    lng = direction == "long"

    # Check each bar closes against our position
    for i in range(bars):
        idx = n - i
        if lng and cl[idx] >= op[idx]:    # bullish bar breaks pressure → reset
            return None
        if not lng and cl[idx] <= op[idx]:
            return None
        # Each individual move must be at least 0.15×ATR (not random doji)
        bar_move = abs(cl[idx] - cl[idx - 1])
        if bar_move < atr14 * 0.15:
            return None

    # Total cumulative move
    total = abs(cl[n] - cl[n - bars])
    if total < atr14 * 0.5:
        return None

    direction_label = "bear" if lng else "bull"
    return (f"Pressure: {bars}× {direction_label} closes, "
            f"total={total:.1f}>{atr14*0.5:.1f}ATR")


def _momentum_reversal(op: list, cl: list, vol: list, rsi: list,
                       n: int, atr14: float, volma20: float,
                       direction: str) -> str | None:
    """
    Detect genuine momentum flip for 15m TF — close-confirmed body + volume.
    Designed to react within 1-2 bars (15-30 min).
    Fires even during spike_guard (real reversal ≠ wick spike).

    Conditions (any one suffices):
      A) Single strong opposing candle: body ≥ 0.8×ATR  AND  vol ≥ 1.5×MA20
      B) Current close breaks 2-bar move ≥ 1.2×ATR (fast sustained pressure)
      C) Single bar body ≥ 1.5×ATR (extreme candle, no vol required)
    RSI momentum confirmation applied to A & B (must be moving against position).
    """
    if n < 2 or atr14 <= 0 or math.isnan(volma20) or volma20 <= 0:
        return None

    lng       = direction == "long"
    body      = abs(cl[n] - op[n])
    vol_ratio = vol[n] / volma20
    rsi_v     = float(rsi[n]) if n < len(rsi) and not math.isnan(float(rsi[n])) else 50.0
    rsi_ok    = (rsi_v < 50) if lng else (rsi_v > 50)  # RSI confirms opposing direction

    # ── A: strong opposing bar + volume (fires in 1 bar / 15 min) ────────
    if lng and cl[n] < op[n] and body >= atr14 * 0.8 and vol_ratio >= 1.5 and rsi_ok:
        return f"MomRev↓ body={body:.1f}>{atr14*0.8:.1f}ATR vol×{vol_ratio:.1f}"
    if not lng and cl[n] > op[n] and body >= atr14 * 0.8 and vol_ratio >= 1.5 and rsi_ok:
        return f"MomRev↑ body={body:.1f}>{atr14*0.8:.1f}ATR vol×{vol_ratio:.1f}"

    # ── B: fast 2-bar sustained move (fires in 2 bars / 30 min) ─────────
    drop_2bar = cl[n - 2] - cl[n]
    rise_2bar = cl[n] - cl[n - 2]
    if lng  and drop_2bar >= atr14 * 1.2 and rsi_ok:
        return f"MomRev↓ 2-bar {drop_2bar:.1f}>1.2×ATR"
    if not lng and rise_2bar >= atr14 * 1.2 and rsi_ok:
        return f"MomRev↑ 2-bar {rise_2bar:.1f}>1.2×ATR"

    # ── C: extreme single bar (no vol needed — size speaks for itself) ───
    if lng and cl[n] < op[n] and body >= atr14 * 1.5:
        return f"MomRev↓ extreme body={body:.1f}>1.5×ATR"
    if not lng and cl[n] > op[n] and body >= atr14 * 1.5:
        return f"MomRev↑ extreme body={body:.1f}>1.5×ATR"

    return None




class SwingReversalPro(BaseStrategy):
    """
    SWING REVERSAL PRO V1 — direction = 'long' | 'short'.
    Create one instance per direction per symbol.
    """

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.direction      = self.params.get("direction", "long")
        self.risk_pct       = float(self.params.get("risk_pct",       0.01))   # 1%
        self.l1_min_score   = int(self.params.get("l1_min_score",       5))
        self.l2_min_pass    = int(self.params.get("l2_min_pass",         5))
        self.sl_atr_min     = float(self.params.get("sl_atr_min",       1.0))
        self.adx_4h_max     = float(self.params.get("adx_4h_max",      35.0))
        self.adx_no_trade   = float(self.params.get("adx_no_trade",    15.0))
        self.atr_min_ratio  = float(self.params.get("atr_min_ratio",    0.8))
        self.mtf_bias_limit = float(self.params.get("mtf_bias_limit",  50.0))
        self.name           = (f"{self.__class__.__name__}_"
                               f"{'L' if self.direction == 'long' else 'S'}")

        # Position state
        self._in_position   = False
        self._pos_side      = self.direction.upper()   # "LONG" | "SHORT"

    # ── Main entry point ──────────────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        mtf  = mtf_candles or {}
        cp   = current_price

        c15m = mtf.get("15m") or candles
        c1h  = mtf.get("1h",  [])
        c4h  = mtf.get("4h",  [])
        n    = len(c15m) - 1

        if n < _WARMUP_15M:
            return Signal(_HOLD, self.symbol, cp, 0,
                          f"[{self.name}] warmup ({n}/{_WARMUP_15M})")

        # ── Compute 15m indicators ────────────────────────────────────────
        cl  = [float(c.close)  for c in c15m]
        hi  = [float(c.high)   for c in c15m]
        lo  = [float(c.low)    for c in c15m]
        op  = [float(c.open)   for c in c15m]
        vol = [float(c.volume) for c in c15m]

        atr14_arr = self.atr(c15m, 14)
        atr14     = float(atr14_arr[n])
        if math.isnan(atr14) or atr14 <= 0:
            atr14 = cp * 0.003

        atr50_arr = self.atr(c15m, 50) if n >= 55 else atr14_arr
        atr50     = float(atr50_arr[n])
        if math.isnan(atr50) or atr50 <= 0:
            atr50 = atr14

        ema50_arr = self.ema(cl, 50)
        ema50     = float(ema50_arr[n])

        rsi14_arr = self.rsi(cl, 14)
        rsi14     = float(rsi14_arr[n])

        adx_arr, _, _ = self.adx(c15m, 14)
        adx14         = float(adx_arr[n])

        hma20_arr = self.hma(cl, 20)

        volma20_arr = self.sma(vol, 20)
        volma20     = float(volma20_arr[n])

        # ── DO NOT TRADE filters ──────────────────────────────────────────
        if not math.isnan(adx14) and adx14 < self.adx_no_trade:
            return Signal(_HOLD, self.symbol, cp, 0,
                          f"[{self.name}] ADX={adx14:.1f}<{self.adx_no_trade}")

        if atr14 < self.atr_min_ratio * atr50:
            return Signal(_HOLD, self.symbol, cp, 0,
                          f"[{self.name}] ATR too low ({atr14:.2f}<{self.atr_min_ratio:.0%}×ATR50)")

        if n >= 20:
            rng20 = max(hi[n - 19: n + 1]) - min(lo[n - 19: n + 1])
            if rng20 < atr14:
                return Signal(_HOLD, self.symbol, cp, 0,
                              f"[{self.name}] Sideways: range={rng20:.2f} < ATR")

        # ── 1H context ───────────────────────────────────────────────────
        n1h       = len(c1h) - 1
        has_1h    = n1h >= _WARMUP_1H
        rsi14_1h  = float("nan")
        atr14_1h  = atr14 * 4
        ema20_1h  = [float("nan")]
        volma_1h  = [float("nan")]

        if has_1h:
            cl1h      = [float(c.close)  for c in c1h]
            vl1h      = [float(c.volume) for c in c1h]
            rsi14_1h  = float(self.rsi(cl1h, 14)[n1h])
            ema20_1h  = self.ema(cl1h, 20)
            atr14_1h_arr = self.atr(c1h, 14)
            atr14_1h  = float(atr14_1h_arr[n1h])
            if math.isnan(atr14_1h) or atr14_1h <= 0:
                atr14_1h = atr14 * 4
            volma_1h  = self.sma(vl1h, 20)

        # ── 4H structure ─────────────────────────────────────────────────
        n4h    = len(c4h) - 1
        has_4h = n4h >= _WARMUP_4H
        adx_4h = float("nan")
        atr14_4h = atr14 * 16
        hi4h = lo4h = []

        if has_4h:
            cl4h  = [float(c.close) for c in c4h]
            hi4h  = [float(c.high)  for c in c4h]
            lo4h  = [float(c.low)   for c in c4h]
            adx_4h_arr, _, _ = self.adx(c4h, 14)
            adx_4h = float(adx_4h_arr[n4h])
            atr14_4h_arr = self.atr(c4h, 14)
            atr14_4h = float(atr14_4h_arr[n4h])
            if math.isnan(atr14_4h) or atr14_4h <= 0:
                atr14_4h = atr14 * 16

        # ── Health score ──────────────────────────────────────────────────
        health_score = int(mtf.get("health_score", 100))

        # MTF Bias
        mtf_bias, _ = self.compute_mtf_bias(c15m, {"1h": c1h, "4h": c4h})

        # ── Early exit (if in position) ───────────────────────────────────
        spike_guard = bool(mtf.get("spike_guard", False))
        if self._in_position:
            exit_reason = self._early_exit(
                cp, cl, hi, lo, op, vol, n, atr14, volma20,
                rsi14, list(rsi14_arr),
                health_score, ema20_1h, n1h, spike_guard)
            if exit_reason:
                self._in_position = False
                return Signal(
                    _SELL, self.symbol, cp, 0,
                    reason=f"[{self.name}] Exit: {exit_reason}",
                    metadata={"position_side": self._pos_side,
                              "action": "close", "exit_reason": exit_reason},
                )
            return Signal(_HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Holding {self._pos_side}")

        # ── Layer 1: Reversal Score ───────────────────────────────────────
        l1, l1_reasons = self._layer1(
            cp, cl, hi, lo, vol, n,
            rsi14, list(rsi14_arr), hma20_arr, adx_arr,
            ema50, atr14, volma20)

        if l1 < self.l1_min_score:
            return Signal(_HOLD, self.symbol, cp, 0,
                          f"[{self.name}] L1={l1}/{self.l1_min_score} "
                          f"({','.join(l1_reasons) or 'none'})")

        # ── Layer 2: Context Filter ───────────────────────────────────────
        l2, l2_reasons = self._layer2(
            cp, rsi14_1h, adx_4h, atr14_4h,
            hi4h, lo4h, n4h, has_4h,
            mtf_bias, ema20_1h, n1h, has_1h,
            atr14_1h, volma_1h)

        if l2 < self.l2_min_pass:
            return Signal(_HOLD, self.symbol, cp, 0,
                          f"[{self.name}] L2={l2}/{self.l2_min_pass}")

        # ── Layer 3: Price Action Trigger ─────────────────────────────────
        trigger, trigger_name, priority = self._layer3(
            cp, cl, hi, lo, op, n, atr14, volma20, vol)

        if not trigger:
            return Signal(_HOLD, self.symbol, cp, 0,
                          f"[{self.name}] L1={l1} L2={l2} waiting trigger")

        # ── Volume confirmation (final gate) ──────────────────────────────
        vol_ok = (not math.isnan(volma20) and volma20 > 0
                  and vol[n] >= volma20 * 1.0)   # at least avg volume

        if not vol_ok:
            return Signal(_HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Volume not confirmed")

        # ── Build signal ──────────────────────────────────────────────────
        sl_p, tp1, tp2, r_dist = self._calc_sltp(
            cp, hi, lo, n, atr14)

        self._in_position = True

        return Signal(
            type=_BUY,
            symbol=self.symbol,
            price=cp,
            amount=0.0,
            confidence=min(0.60 + l1 * 0.04 + (1 / priority) * 0.08, 0.95),
            reason=(f"[{self.name}] L1={l1}/7 L2={l2}/6 "
                    f"Trigger={trigger_name} MTFbias={mtf_bias:+.0f}"),
            metadata={
                "position_side": self._pos_side,
                "action":        "open",
                "stop_loss":     sl_p,
                "take_profit":   tp2,
                "tp1":           tp1,
                "r_dist":        round(r_dist, 4),
                "atr":           round(atr14, 4),
                "l1_score":      l1,
                "l2_score":      l2,
                "trigger":       trigger_name,
                "priority":      priority,
                "mtf_bias":      round(mtf_bias, 1),
            },
        )

    # ── Layer 1 ───────────────────────────────────────────────────────────────

    def _layer1(self, cp, cl, hi, lo, vol, n,
                rsi14, rsi14_arr, hma20, adx_arr,
                ema50, atr14, volma20) -> tuple[int, list]:
        score  = 0
        passed = []
        lng    = self.direction == "long"

        # [1] RSI threshold
        if lng and not math.isnan(rsi14) and rsi14 < 35:
            score += 1; passed.append("RSI<35")
        elif not lng and not math.isnan(rsi14) and rsi14 > 65:
            score += 1; passed.append("RSI>65")

        # [2] RSI Divergence
        div = _rsi_divergence(cl, rsi14_arr)
        if lng and div == "bullish":
            score += 1; passed.append("BullDiv")
        elif not lng and div == "bearish":
            score += 1; passed.append("BearDiv")

        # [3] HMA20 slope
        slope = _hma_slope(hma20, n)
        if lng and slope >= 0:
            score += 1; passed.append("HMAflat+")
        elif not lng and slope <= 0:
            score += 1; passed.append("HMAflat-")

        # [4] ADX rollover
        if _adx_rollover(adx_arr, n):
            score += 1; passed.append("ADXroll")

        # [5] Volume spike
        if (not math.isnan(volma20) and volma20 > 0
                and vol[n] > volma20 * 1.5):
            score += 1; passed.append("VolSpike")

        # [6] Liquidity sweep
        if lng and _liquidity_sweep_low(hi, lo, cl, n):
            score += 1; passed.append("SweepLow")
        elif not lng and _liquidity_sweep_high(hi, lo, cl, n):
            score += 1; passed.append("SweepHigh")

        # [7] Distance extension from EMA50
        if not math.isnan(ema50):
            dist = (ema50 - cp) if lng else (cp - ema50)
            if dist > 1.5 * atr14:
                score += 1; passed.append("Extension")

        return score, passed

    # ── Layer 2 ───────────────────────────────────────────────────────────────

    def _layer2(self, cp, rsi14_1h, adx_4h, atr14_4h,
                hi4h, lo4h, n4h, has_4h,
                mtf_bias, ema20_1h, n1h, has_1h,
                atr14_1h, volma_1h) -> tuple[int, list]:
        score  = 0
        passed = []
        lng    = self.direction == "long"

        # [1] 4H ADX < 35
        if math.isnan(adx_4h) or adx_4h < self.adx_4h_max:
            score += 1; passed.append("4HADX<35")

        # [2] 1H RSI gating
        if lng:
            if math.isnan(rsi14_1h) or rsi14_1h > 30:
                score += 1; passed.append("1HRSI>30")
        else:
            if math.isnan(rsi14_1h) or rsi14_1h < 70:
                score += 1; passed.append("1HRSI<70")

        # [3] 1H Volume trend not declining
        if has_1h and n1h >= 3:
            if _vol_not_declining(volma_1h, n1h):
                score += 1; passed.append("VolOK")
        else:
            score += 1; passed.append("VolOK?")  # no data → pass

        # [4] Near 4H S/R zone
        if has_4h:
            if lng and _near_support(cp, lo4h, n4h, atr14_4h):
                score += 1; passed.append("NearSupp")
            elif not lng and _near_resistance(cp, hi4h, n4h, atr14_4h):
                score += 1; passed.append("NearRes")
        else:
            score += 1; passed.append("SRzone?")

        # [5] MTF Bias within permitted range
        if lng and mtf_bias > -self.mtf_bias_limit:
            score += 1; passed.append(f"MTFbias{mtf_bias:+.0f}>-{self.mtf_bias_limit:.0f}")
        elif not lng and mtf_bias < self.mtf_bias_limit:
            score += 1; passed.append(f"MTFbias{mtf_bias:+.0f}<+{self.mtf_bias_limit:.0f}")

        # [6] EMA20 1H slope not extreme
        if has_1h and n1h >= 4:
            slope_1h = _ema20_slope(ema20_1h, n1h)
            slope_per_atr = slope_1h / max(atr14_1h, 1e-10)
            ok = (slope_per_atr > -0.15) if lng else (slope_per_atr < 0.15)
            if ok:
                score += 1; passed.append("EMAslope")
        else:
            score += 1; passed.append("EMAslope?")

        return score, passed

    # ── Layer 3 ───────────────────────────────────────────────────────────────

    def _layer3(self, cp, cl, hi, lo, op, n, atr14, volma20,
                vol) -> tuple[bool, str, int]:
        """Returns (triggered, name, priority). Priority 1=best."""
        lng = self.direction == "long"

        # Priority 1: CHOCH + Divergence (handled via L1 divergence already)
        if lng and _choch_bull(hi, cl, n):
            return True, "CHOCH_bull", 1
        if not lng and _choch_bear(lo, cl, n):
            return True, "CHOCH_bear", 1

        # Priority 1: BOS
        if lng and _bos_bull(hi, cl, n):
            return True, "BOS_bull", 1
        if not lng and _bos_bear(lo, cl, n):
            return True, "BOS_bear", 1

        # Priority 2: Double Bottom/Top + volume
        vol_spike = (not math.isnan(volma20) and volma20 > 0
                     and vol[n] > volma20 * 1.5)
        if lng and _double_bottom(lo, cl, n) and vol_spike:
            return True, "DblBottom", 2
        if not lng and _double_top(hi, cl, n) and vol_spike:
            return True, "DblTop", 2

        # Priority 3: Hammer / Engulfing
        if lng and _hammer(op[n], hi[n], lo[n], cl[n], atr14):
            return True, "Hammer", 3
        if not lng and _shooting_star(op[n], hi[n], lo[n], cl[n], atr14):
            return True, "ShootStar", 3
        if lng and _bull_engulf(op, cl, n):
            return True, "BullEngulf", 3
        if not lng and _bear_engulf(op, cl, n):
            return True, "BearEngulf", 3

        # Priority 4: V-Reversal
        if lng and _v_reversal_long(cl, lo, n, atr14):
            return True, "V-Rev_long", 4
        if not lng and _v_reversal_short(cl, hi, n, atr14):
            return True, "V-Rev_short", 4

        return False, "", 0

    # ── Early exit ────────────────────────────────────────────────────────────

    def _early_exit(self, cp, cl, hi, lo, op, vol, n, atr14, volma20,
                    rsi14, rsi14_arr,
                    health_score, ema20_1h, n1h,
                    spike_guard: bool = False) -> str | None:
        lng = self.direction == "long"

        # [1] Health score emergency — always fires, no override
        if health_score < 25:
            return f"health={health_score}<25"

        # [2] Momentum reversal — strong single candle + volume (1-2 bars / 15-30 min)
        #     Fires even during spike_guard: large body = real move, not a wick.
        mom_rev = _momentum_reversal(op, cl, vol, rsi14_arr, n, atr14, volma20,
                                     self.direction)
        if mom_rev:
            return mom_rev

        # [3] Candle pressure — 2 consecutive opposing closes (30-45 min early warning).
        #     Catches gradual 3-4 bar builds before they reach SL.
        #     Fires even during spike_guard (sustained closes ≠ wick).
        pressure = _candle_pressure(op, cl, hi, lo, n, atr14, self.direction, bars=2)
        if pressure:
            return pressure

        # [4] During V-spike guard: suppress remaining technical exits (CHOCH/RSI/div)
        #     so wick-stop-hunt moves don't force an exit at the worst price.
        if spike_guard:
            return None

        if lng:
            if not math.isnan(rsi14) and rsi14 > 75:
                return f"RSI={rsi14:.1f}>75"
            if _choch_bear(lo, cl, n):
                return "BearCHOCH"
            if _rsi_divergence(cl, rsi14_arr) == "bearish":
                return "NewBearDiv"
        else:
            if not math.isnan(rsi14) and rsi14 < 25:
                return f"RSI={rsi14:.1f}<25"
            if _choch_bull(hi, cl, n):
                return "BullCHOCH"
            if _rsi_divergence(cl, rsi14_arr) == "bullish":
                return "NewBullDiv"
        return None

    # ── SL / TP ───────────────────────────────────────────────────────────────

    def _calc_sltp(self, cp, hi, lo, n,
                   atr14) -> tuple[float, float, float, float]:
        """
        SL = wider of (pattern extreme | 1.0×ATR14)
        TP1 = 0.8R  (70% close)
        TP2 = 1.5R  (remaining)
        """
        lng = self.direction == "long"

        # Pattern extreme over last 5 bars
        lookback = min(5, n)
        if lng:
            pattern_extreme = min(lo[n - lookback: n + 1])
            sl_pattern_dist = cp - pattern_extreme
            sl_atr_dist     = self.sl_atr_min * atr14
            sl_dist         = max(sl_pattern_dist, sl_atr_dist)
            sl_p            = round(cp - sl_dist, 4)
            tp1             = round(cp + 0.8 * sl_dist, 4)
            tp2             = round(cp + 1.5 * sl_dist, 4)
        else:
            pattern_extreme = max(hi[n - lookback: n + 1])
            sl_pattern_dist = pattern_extreme - cp
            sl_atr_dist     = self.sl_atr_min * atr14
            sl_dist         = max(sl_pattern_dist, sl_atr_dist)
            sl_p            = round(cp + sl_dist, 4)
            tp1             = round(cp - 0.8 * sl_dist, 4)
            tp2             = round(cp - 1.5 * sl_dist, 4)

        return sl_p, tp1, tp2, sl_dist
