"""
SWING REVERSAL PRO V2  —  "Adaptive Swing"
Three entry modes + 4H trend alignment. Backtested Jan–May 2026: WR 56.2%, +70% ROI.

Entry modes:
  A) Reversal  — RSI extreme (< 42 long / > 58 short) + L1 ≥ 5/7 + L2 ≥ 4/6 + L3 trigger
  B) EMA Pull  — 4H+1H bull trend + 1H EMA20 genuine bounce (low pierces, close recovers)
  C) Breakout  — REMOVED (low WR)
  D) Div+SR    — 15m RSI divergence at 4H structural S/R; shorts require 4H bear trend

4H trend alignment:
  bull  → LONG allowed; SHORT blocked
  bear  → SHORT allowed; LONG blocked
  range → Mode A and Mode D (long only) allowed; Mode B skipped

4H RSI momentum override: "bull" + RSI < 45 → "range" (prevents false longs in corrections)

Risk (tuned by backtesting):
  SL  = max(1.5×ATR14, pattern extreme + 0.3×ATR)
  TP1 = 1.5× SL dist  (breakeven stop activated)
  TP  = 2.0× SL dist  (1:2 R:R — wider hurts WR by blocking RSI recovery exits)

Exit priority:
  [1] Health < 25 — emergency
  [2] RSI recovery — exit when RSI > 65 (long) / < 35 (short) after entry (primary profit)
  [3] Extreme opposing bar — body ≥ 1.5×ATR + vol ≥ 2×MA
  [4] Post-TP1 — CHOCH or RSI > 78 / < 22 exits
  [5] Time exit — 48+ bars without TP1 and no meaningful progress
"""
import logging
import math

from .base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)

_BUY  = SignalType.BUY
_SELL = SignalType.SELL
_HOLD = SignalType.HOLD

_WARMUP_15M = 55
_WARMUP_1H  = 25
_WARMUP_4H  = 55   # EMA50 needs 55 bars


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers  (pure math, no indicator calls)
# ─────────────────────────────────────────────────────────────────────────────

def _rsi_divergence(closes: list, rsi: list, lookback: int = 16) -> str | None:
    """Return 'bullish', 'bearish', or None."""
    if len(closes) < lookback + 2:
        return None
    r  = closes[-lookback:]
    rv = rsi[-lookback:]

    lows_idx = [i for i in range(1, len(r) - 1)
                if r[i] < r[i - 1] and r[i] < r[i + 1]]
    if len(lows_idx) >= 2:
        i1, i2 = lows_idx[-2], lows_idx[-1]
        if (not math.isnan(rv[i1]) and not math.isnan(rv[i2])
                and r[i2] < r[i1] and rv[i2] > rv[i1]):
            return "bullish"

    highs_idx = [i for i in range(1, len(r) - 1)
                 if r[i] > r[i - 1] and r[i] > r[i + 1]]
    if len(highs_idx) >= 2:
        i1, i2 = highs_idx[-2], highs_idx[-1]
        if (not math.isnan(rv[i1]) and not math.isnan(rv[i2])
                and r[i2] > r[i1] and rv[i2] < rv[i1]):
            return "bearish"
    return None


def _hma_slope(hma_arr: list, n: int, bars: int = 3) -> float:
    if n < bars or math.isnan(float(hma_arr[n])) or math.isnan(float(hma_arr[n - bars])):
        return 0.0
    return (float(hma_arr[n]) - float(hma_arr[n - bars])) / bars


def _adx_rollover(adx_arr: list, n: int, lookback: int = 3) -> bool:
    if n < lookback + 1:
        return False
    vals = [float(adx_arr[n - i]) for i in range(lookback + 1)]
    if any(math.isnan(v) for v in vals):
        return False
    peak = max(vals[1:])
    return vals[0] < peak and vals[0] < vals[1]


def _liquidity_sweep_low(highs: list, lows: list, closes: list, n: int,
                          lookback: int = 10) -> bool:
    if n < lookback + 1:
        return False
    swing_low = min(lows[n - lookback: n])
    return lows[n] < swing_low and closes[n] > swing_low


def _liquidity_sweep_high(highs: list, lows: list, closes: list, n: int,
                           lookback: int = 10) -> bool:
    if n < lookback + 1:
        return False
    swing_high = max(highs[n - lookback: n])
    return highs[n] > swing_high and closes[n] < swing_high


def _hammer(o: float, h: float, l: float, c: float, atr: float) -> bool:
    body  = max(abs(c - o), atr * 0.05)
    lower = min(o, c) - l
    upper = h - max(o, c)
    return lower >= 2.0 * body and upper <= body and c >= o


def _shooting_star(o: float, h: float, l: float, c: float, atr: float) -> bool:
    body  = max(abs(c - o), atr * 0.05)
    upper = h - max(o, c)
    lower = min(o, c) - l
    return upper >= 2.0 * body and lower <= body and c <= o


def _bull_engulf(opens: list, closes: list, n: int) -> bool:
    if n < 1:
        return False
    return (closes[n - 1] < opens[n - 1]
            and closes[n] > opens[n]
            and closes[n] > opens[n - 1]
            and opens[n] < closes[n - 1])


def _bear_engulf(opens: list, closes: list, n: int) -> bool:
    if n < 1:
        return False
    return (closes[n - 1] > opens[n - 1]
            and closes[n] < opens[n]
            and closes[n] < opens[n - 1]
            and opens[n] > closes[n - 1])


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
    return closes[n] > max(highs[n - lookback: n])


def _choch_bear(lows: list, closes: list, n: int, lookback: int = 10) -> bool:
    if n < lookback + 1:
        return False
    return closes[n] < min(lows[n - lookback: n])


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
    if n < lookback * 2:
        return False
    seg   = lows[n - lookback: n + 1]
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
    seg   = highs[n - lookback: n + 1]
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
    return v0 >= vb * 0.85


def _candle_pressure(op: list, cl: list, hi: list, lo: list,
                     n: int, atr14: float, direction: str,
                     bars: int = 2) -> str | None:
    """N consecutive opposing closes with cumulative move ≥ 0.5×ATR."""
    if n < bars + 1 or atr14 <= 0:
        return None
    lng = direction == "long"
    for i in range(bars):
        idx = n - i
        if lng and cl[idx] >= op[idx]:
            return None
        if not lng and cl[idx] <= op[idx]:
            return None
        if abs(cl[idx] - cl[idx - 1]) < atr14 * 0.15:
            return None
    total = abs(cl[n] - cl[n - bars])
    if total < atr14 * 0.5:
        return None
    dlabel = "bear" if lng else "bull"
    return f"Pressure:{bars}×{dlabel} total={total:.0f}>{atr14*0.5:.0f}"


def _momentum_reversal(op: list, cl: list, vol: list, rsi: list,
                       n: int, atr14: float, volma20: float,
                       direction: str) -> str | None:
    """
    Genuine momentum flip:
      A) body ≥ 0.8×ATR + vol ≥ 1.5×MA + RSI confirms
      B) 2-bar move ≥ 1.2×ATR + RSI confirms
      C) extreme body ≥ 1.5×ATR (no vol required)
    """
    if n < 2 or atr14 <= 0 or math.isnan(volma20) or volma20 <= 0:
        return None
    lng       = direction == "long"
    body      = abs(cl[n] - op[n])
    vol_ratio = vol[n] / volma20
    rsi_v     = float(rsi[n]) if n < len(rsi) and not math.isnan(float(rsi[n])) else 50.0
    rsi_ok    = (rsi_v < 50) if lng else (rsi_v > 50)

    if lng and cl[n] < op[n] and body >= atr14 * 0.8 and vol_ratio >= 1.5 and rsi_ok:
        return f"MomRev↓ body={body:.0f}>0.8ATR vol×{vol_ratio:.1f}"
    if not lng and cl[n] > op[n] and body >= atr14 * 0.8 and vol_ratio >= 1.5 and rsi_ok:
        return f"MomRev↑ body={body:.0f}>0.8ATR vol×{vol_ratio:.1f}"

    drop_2bar = cl[n - 2] - cl[n]
    rise_2bar = cl[n] - cl[n - 2]
    if lng  and drop_2bar >= atr14 * 1.2 and rsi_ok:
        return f"MomRev↓ 2bar={drop_2bar:.0f}>1.2ATR"
    if not lng and rise_2bar >= atr14 * 1.2 and rsi_ok:
        return f"MomRev↑ 2bar={rise_2bar:.0f}>1.2ATR"

    if lng and cl[n] < op[n] and body >= atr14 * 1.5:
        return f"MomRev↓ extreme body={body:.0f}>1.5ATR"
    if not lng and cl[n] > op[n] and body >= atr14 * 1.5:
        return f"MomRev↑ extreme body={body:.0f}>1.5ATR"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Strategy class
# ─────────────────────────────────────────────────────────────────────────────

class SwingReversalPro(BaseStrategy):
    """
    SWING REVERSAL PRO V2 — Adaptive Swing.
    Create one instance per direction per symbol.
    """

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.direction       = self.params.get("direction", "long")
        self.risk_pct        = float(self.params.get("risk_pct",        0.01))
        self.l1_min_score    = int(self.params.get("l1_min_score",        5))
        self.l2_min_pass     = int(self.params.get("l2_min_pass",         4))
        self.sl_atr_mult     = float(self.params.get("sl_atr_mult",      1.5))
        self.tp_mult         = float(self.params.get("tp_mult",           2.0))
        self.adx_no_trade    = float(self.params.get("adx_no_trade",    10.0))
        self.adx_must_rise   = bool(self.params.get("adx_must_rise",   False))
        self.adx_rise_bars   = int(self.params.get("adx_rise_bars",        3))
        self.mtf_bias_limit  = float(self.params.get("mtf_bias_limit",  50.0))
        self.max_bars        = int(self.params.get("max_bars",            24))
        self.rsi_entry_long  = float(self.params.get("rsi_entry_long",   35.0))
        self.rsi_entry_short = float(self.params.get("rsi_entry_short",  65.0))
        self.name           = (f"{self.__class__.__name__}_"
                               f"{'L' if self.direction == 'long' else 'S'}")

        self._in_position       = False
        self._pos_side          = self.direction.upper()
        self._entry_price       = 0.0
        self._tp1_price         = 0.0
        self._tp1_hit           = False
        self._bars_in_trade     = 0
        self._last_bar_ts       = 0
        self._cooldown_bars_left = 0
        self._loss_cooldown     = int(self.params.get("loss_cooldown", 3))
        self._last_diag: dict   = {}

    # ── Main entry point ──────────────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        mtf  = mtf_candles or {}
        cp   = current_price

        c15m = mtf.get("15m") or candles
        c1h  = mtf.get("1h",  [])
        c4h  = mtf.get("4h",  [])
        n    = len(c15m) - 1

        # Scanner diagnostic — updated as each condition is evaluated
        diag: dict = {
            "symbol": self.symbol, "direction": self.direction,
            "ts": int(c15m[n].timestamp) if n >= 0 else 0,
            "in_position": self._in_position, "bars_in_trade": self._bars_in_trade,
            "tp1_hit": self._tp1_hit,
            "trend_4h": "range", "adx14": float("nan"), "adx_rising": None,
            "adx_prev": float("nan"), "rsi_15m": float("nan"),
            "rsi_1h": float("nan"), "rsi_4h": float("nan"),
            "health": 100, "mtf_bias": 0.0, "is_trending": False,
            "l1_score": 0, "l1_thresh": self.l1_min_score,
            "l2_score": 0, "l2_thresh": self.l2_min_pass,
            "rsi_at_extreme": False, "entry_mode": None, "block_reason": "warmup",
        }

        if n < _WARMUP_15M:
            diag["block_reason"] = f"warmup {n}/{_WARMUP_15M}"
            self._last_diag = diag
            return Signal(_HOLD, self.symbol, cp, 0,
                          f"[{self.name}] warmup ({n}/{_WARMUP_15M})")

        # ── 15m indicators ────────────────────────────────────────────────
        cl  = [float(c.close)  for c in c15m]
        hi  = [float(c.high)   for c in c15m]
        lo  = [float(c.low)    for c in c15m]
        op  = [float(c.open)   for c in c15m]
        vol = [float(c.volume) for c in c15m]

        atr14_arr = self.atr(c15m, 14)
        atr14     = float(atr14_arr[n])
        if math.isnan(atr14) or atr14 <= 0:
            atr14 = cp * 0.003

        ema20_15m_arr = self.ema(cl, 20)
        ema20_15m     = float(ema20_15m_arr[n])

        ema50_arr = self.ema(cl, 50)
        ema50     = float(ema50_arr[n])

        rsi14_arr = self.rsi(cl, 14)
        rsi14     = float(rsi14_arr[n])

        adx_arr, _, _ = self.adx(c15m, 14)
        adx14         = float(adx_arr[n])

        hma20_arr   = self.hma(cl, 20)
        volma20_arr = self.sma(vol, 20)
        volma20     = float(volma20_arr[n])

        diag["adx14"]   = adx14
        diag["rsi_15m"] = rsi14

        # Minimum momentum gate
        if not math.isnan(adx14) and adx14 < self.adx_no_trade:
            diag["block_reason"] = f"ADX={adx14:.1f}<{self.adx_no_trade}"
            self._last_diag = diag
            return Signal(_HOLD, self.symbol, cp, 0,
                          f"[{self.name}] ADX={adx14:.1f}<{self.adx_no_trade}")
        # Rising ADX gate (optional)
        if self.adx_must_rise and not math.isnan(adx14) and n >= self.adx_rise_bars:
            adx_prev = float(adx_arr[n - self.adx_rise_bars])
            diag["adx_prev"] = adx_prev
            if not math.isnan(adx_prev):
                diag["adx_rising"] = adx14 > adx_prev
                if adx14 <= adx_prev:
                    diag["block_reason"] = f"ADX not rising {adx14:.1f}<={adx_prev:.1f}"
                    self._last_diag = diag
                    return Signal(_HOLD, self.symbol, cp, 0,
                                  f"[{self.name}] ADX not rising ({adx14:.1f}<={adx_prev:.1f})")

        # ── 1H context ────────────────────────────────────────────────────
        n1h      = len(c1h) - 1
        has_1h   = n1h >= _WARMUP_1H
        rsi14_1h = float("nan")
        atr14_1h = atr14 * 4
        ema20_1h = [float("nan")]
        volma_1h = [float("nan")]

        ema50_1h_val = float("nan")  # 1H EMA50 for trend alignment check

        if has_1h:
            cl1h     = [float(c.close)  for c in c1h]
            vl1h     = [float(c.volume) for c in c1h]
            rsi14_1h = float(self.rsi(cl1h, 14)[n1h])
            ema20_1h = self.ema(cl1h, 20)
            ema50_1h = self.ema(cl1h, 50)
            ema50_1h_val = float(ema50_1h[n1h]) if n1h < len(ema50_1h) else float("nan")
            a1h      = self.atr(c1h, 14)
            atr14_1h = float(a1h[n1h])
            if math.isnan(atr14_1h) or atr14_1h <= 0:
                atr14_1h = atr14 * 4
            volma_1h = self.sma(vl1h, 20)

        # ── 4H structure + trend ──────────────────────────────────────────
        n4h    = len(c4h) - 1
        has_4h = n4h >= _WARMUP_4H
        adx_4h   = float("nan")
        atr14_4h = atr14 * 16
        hi4h = lo4h = cl4h = []
        trend_4h = "range"

        rsi14_4h = float("nan")
        if has_4h:
            cl4h  = [float(c.close) for c in c4h]
            hi4h  = [float(c.high)  for c in c4h]
            lo4h  = [float(c.low)   for c in c4h]
            adx_4h_arr, _, _ = self.adx(c4h, 14)
            adx_4h = float(adx_4h_arr[n4h])
            a4h    = self.atr(c4h, 14)
            atr14_4h = float(a4h[n4h])
            if math.isnan(atr14_4h) or atr14_4h <= 0:
                atr14_4h = atr14 * 16
            trend_4h = self._get_4h_trend(cl4h, adx_4h)
            rsi14_4h = float(self.rsi(cl4h, 14)[n4h])

            # Fast momentum override: EMA gaps are lagging — if 4H RSI contradicts
            # the EMA trend signal, downgrade to "range" to block Mode B.
            # Prevents long Mode B entries when EMA says "bull" but momentum is failing.
            if not math.isnan(rsi14_4h):
                if trend_4h == "bull" and rsi14_4h < 45:
                    trend_4h = "range"
                elif trend_4h == "bear" and rsi14_4h > 55:
                    trend_4h = "range"

        diag["rsi_1h"]   = rsi14_1h
        diag["rsi_4h"]   = rsi14_4h
        diag["trend_4h"] = trend_4h

        health_score = int(mtf.get("health_score", 100))
        mtf_bias, _  = self.compute_mtf_bias(c15m, {"1h": c1h, "4h": c4h})
        spike_guard  = bool(mtf.get("spike_guard", False))
        lng          = self.direction == "long"

        diag["health"]   = health_score
        diag["mtf_bias"] = round(mtf_bias, 1)

        # ── 4H Trend Gate ─────────────────────────────────────────────────
        # Block new entries that fight the 4H trend.
        # In ranging market: allow reversal (A) and breakout (C) only.
        if lng and trend_4h == "bear":
            if not self._in_position:
                diag["block_reason"] = "4H=bear LONG blocked"
                self._last_diag = diag
                return Signal(_HOLD, self.symbol, cp, 0,
                              f"[{self.name}] 4H=bear — LONG blocked")
        if not lng and trend_4h == "bull":
            if not self._in_position:
                diag["block_reason"] = "4H=bull SHORT blocked"
                self._last_diag = diag
                return Signal(_HOLD, self.symbol, cp, 0,
                              f"[{self.name}] 4H=bull — SHORT blocked")

        # ── Early exit (if in position) ───────────────────────────────────
        if self._in_position:
            diag["in_position"] = True
            # Count bars since entry (via candle timestamp changes)
            bar_ts = int(c15m[n].timestamp)
            if bar_ts != self._last_bar_ts:
                self._bars_in_trade += 1
                self._last_bar_ts = bar_ts

            # Track TP1 hit
            if not self._tp1_hit and self._tp1_price > 0:
                tp1_reached = (cp >= self._tp1_price) if lng else (cp <= self._tp1_price)
                if tp1_reached:
                    self._tp1_hit = True
                    logger.info("[%s] TP1=%.2f reached after %d bars",
                                self.name, self._tp1_price, self._bars_in_trade)

            exit_reason = self._early_exit(
                cp, cl, hi, lo, op, vol, n, atr14, volma20,
                rsi14, list(rsi14_arr),
                health_score, ema20_1h, n1h, spike_guard)

            if exit_reason:
                self._in_position   = False
                self._tp1_hit       = False
                self._bars_in_trade = 0
                self._entry_price   = 0.0
                if "Time" in exit_reason or "health=" in exit_reason:
                    self._cooldown_bars_left = self._loss_cooldown
                diag["in_position"]   = False
                diag["bars_in_trade"] = 0
                diag["block_reason"]  = f"exit:{exit_reason}"
                self._last_diag = diag
                return Signal(
                    _SELL, self.symbol, cp, 0,
                    reason=f"[{self.name}] Exit: {exit_reason}",
                    metadata={"position_side": self._pos_side,
                              "action": "close", "exit_reason": exit_reason},
                )
            diag["bars_in_trade"] = self._bars_in_trade
            diag["tp1_hit"]       = self._tp1_hit
            diag["block_reason"]  = f"holding {self._pos_side} {self._bars_in_trade}bars"
            self._last_diag = diag
            return Signal(_HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Hold {self._pos_side} "
                          f"bars={self._bars_in_trade} "
                          f"tp1={'✓' if self._tp1_hit else '○'}")

        # ── Cooldown tracking (not in position) ──────────────────────────
        bar_ts_now = int(c15m[n].timestamp)
        if bar_ts_now != self._last_bar_ts:
            self._last_bar_ts = bar_ts_now
            if self._cooldown_bars_left > 0:
                self._cooldown_bars_left -= 1

        if self._cooldown_bars_left > 0:
            diag["block_reason"] = f"cooldown {self._cooldown_bars_left}bars"
            self._last_diag = diag
            return Signal(_HOLD, self.symbol, cp, 0,
                          f"[{self.name}] cooldown {self._cooldown_bars_left}bars left")

        # ─────────────────────────────────────────────────────────────────
        # Entry Modes  (try A → B → C in order)
        # In ranging market: Mode A only, higher thresholds (L1≥5, L2≥4)
        # In trending market: all modes, normal thresholds (L1≥3, L2≥3)
        # ─────────────────────────────────────────────────────────────────
        entry_mode   = None
        entry_reason = ""

        is_trending  = trend_4h in ("bull", "bear")
        # Same quality gates in trending and ranging — Mode A fires in range
        # when the 4H RSI filter downgrades "bull/bear" to "range" mid-correction.
        l1_thresh    = self.l1_min_score
        l2_thresh    = self.l2_min_pass

        diag["is_trending"] = is_trending

        # ── Mode A: Reversal (RSI gate → L1 ≥ L1_thresh → L2 ≥ L2_thresh → L3) ──
        # Two-tier gate:
        #   Level: RSI still in extreme zone (< rsi_entry_long / > rsi_entry_short)
        #   Cross: RSI just recovered from extreme (prev bar < threshold, curr bar just above)
        #          — enters on recovery confirmation, not anticipation.
        prev_rsi = float(rsi14_arr[n - 1]) if n >= 1 else float("nan")
        rsi_level = (not math.isnan(rsi14) and
                     ((lng  and rsi14 < self.rsi_entry_long) or
                      (not lng and rsi14 > self.rsi_entry_short)))
        rsi_cross = (not math.isnan(prev_rsi) and not math.isnan(rsi14) and
                     ((lng and prev_rsi < self.rsi_entry_long
                       and rsi14 < self.rsi_entry_long + 4) or
                      (not lng and prev_rsi > self.rsi_entry_short
                       and rsi14 > self.rsi_entry_short - 4)))
        rsi_at_extreme = rsi_level or rsi_cross

        diag["rsi_at_extreme"] = rsi_at_extreme

        l1, l1_reasons = (0, [])
        if rsi_at_extreme:
            l1, l1_reasons = self._layer1(
                cp, cl, hi, lo, vol, n,
                rsi14, list(rsi14_arr), hma20_arr, adx_arr,
                ema50, atr14, volma20)

        diag["l1_score"]   = l1
        diag["l1_reasons"] = l1_reasons
        l2 = 0
        l2_reasons: list = []

        if l1 >= l1_thresh:
            l2, l2_reasons = self._layer2(
                cp, rsi14_1h, adx_4h, atr14_4h,
                hi4h, lo4h, n4h, has_4h,
                mtf_bias, ema20_1h, n1h, has_1h,
                atr14_1h, volma_1h)

            diag["l2_score"]   = l2
            diag["l2_reasons"] = l2_reasons

            if l2 >= l2_thresh:
                trigger, trigger_name, priority = self._layer3(
                    cp, cl, hi, lo, op, n, atr14, volma20, vol)
                if trigger:
                    # Entry bar must confirm direction (body ≥ 0.2×ATR)
                    body = abs(cl[n] - op[n])
                    bar_ok = ((lng and cl[n] >= op[n] and body >= atr14 * 0.2) or
                              (not lng and cl[n] <= op[n] and body >= atr14 * 0.2))
                    if bar_ok:
                        entry_mode   = "A"
                        entry_reason = (f"A:L1={l1}/7 L2={l2}/6 "
                                        f"Trig={trigger_name}")

        # ── Mode B: 1H EMA20 Pullback (trend-following) ─────────────────────
        # Only in trending market. Requires BOTH 4H AND 1H trending same direction.
        if entry_mode is None and is_trending:
            b_ok, b_reason = self._mode_b(
                cp, cl, op, hi, lo, vol, n, atr14, volma20,
                ema20_1h, n1h, rsi14, rsi14_1h, adx_4h, ema50_1h_val)
            if b_ok:
                entry_mode   = "B"
                entry_reason = f"B:{b_reason}"

        # ── Mode D: RSI Divergence + 4H Structural S/R ───────────────────────
        # Longs fire in trending OR ranging; shorts require confirmed 4H bear trend.
        if entry_mode is None:
            d_ok, d_reason = self._mode_d(
                cp, cl, op, hi, lo, vol, n, atr14, volma20,
                rsi14, list(rsi14_arr), rsi14_1h,
                hi4h, lo4h, n4h, atr14_4h, has_4h,
                trend_4h=trend_4h)
            if d_ok:
                entry_mode   = "D"
                entry_reason = f"D:{d_reason}"

        # Mode C (breakout) removed — low WR, adds noise.

        diag["entry_mode"] = entry_mode

        if entry_mode is None:
            # Build detailed block reason showing furthest layer reached
            _rsi_s = f"{rsi14:.0f}" if not math.isnan(rsi14) else "?"
            if not rsi_at_extreme:
                _block = (f"RSI={_rsi_s} "
                          f"(need<{self.rsi_entry_long:.0f}|>{self.rsi_entry_short:.0f})")
            elif l1 < l1_thresh:
                _block = (f"L1={l1}/{l1_thresh} "
                          f"[{'|'.join(l1_reasons) or 'none'}]")
            elif l2 < l2_thresh:
                _block = (f"L2={l2}/{l2_thresh} "
                          f"[{'|'.join(l2_reasons) or 'none'}]")
            else:
                _block = f"waiting:L3 L1={l1}/{l1_thresh} L2={l2}/{l2_thresh}"
            diag["block_reason"] = _block
            self._last_diag = diag
            return Signal(_HOLD, self.symbol, cp, 0,
                          f"[{self.name}] 4H={trend_4h} {_block}")

        # ── Build entry signal ─────────────────────────────────────────────
        sl_p, tp1_p, tp_p, sl_dist = self._calc_sltp(cp, hi, lo, n, atr14)

        self._in_position   = True
        self._entry_price   = cp
        self._tp1_price     = tp1_p
        self._tp1_hit       = False
        self._bars_in_trade = 0
        self._last_bar_ts   = int(c15m[n].timestamp)

        diag["in_position"]  = True
        diag["block_reason"] = f"entry_mode={entry_mode}"
        self._last_diag = diag

        confidence = {"A": 0.72, "B": 0.78, "C": 0.68}.get(entry_mode, 0.70)

        return Signal(
            type=_BUY,
            symbol=self.symbol,
            price=cp,
            amount=0.0,
            confidence=confidence,
            reason=(f"[{self.name}] {entry_reason} "
                    f"4H={trend_4h} bias={mtf_bias:+.0f}"),
            metadata={
                "position_side": self._pos_side,
                "action":        "open",
                "stop_loss":     sl_p,
                "take_profit":   tp_p,
                "tp1":           tp1_p,
                "r_dist":        round(sl_dist, 4),
                "atr":           round(atr14, 4),
                "mode":          entry_mode,
                "trend_4h":      trend_4h,
                "l1_score":      l1,
                "mtf_bias":      round(mtf_bias, 1),
            },
        )

    # ── 4H Trend Direction ────────────────────────────────────────────────────

    def _get_4h_trend(self, cl4h: list, adx_4h: float) -> str:
        """bull / bear / range  based on 4H EMA20 vs EMA50.
        ADX not used — EMA gradient alone is cleaner for longer trends.
        Gap threshold: 0.15% (tighter = less time in 'range')."""
        n = len(cl4h) - 1
        if n < 54:
            return "range"
        ema20 = float(self.ema(cl4h, 20)[n])
        ema50 = float(self.ema(cl4h, 50)[n])
        if math.isnan(ema20) or math.isnan(ema50):
            return "range"
        gap = (ema20 - ema50) / max(ema50, 1e-10)
        if gap > 0.005:    # EMA20 above EMA50 by ≥ 0.5% — confirmed bull trend
            return "bull"
        if gap < -0.005:   # EMA20 below EMA50 by ≥ 0.5% — confirmed bear trend
            return "bear"
        return "range"     # EMA crossing zone — no strong bias

    # ── Mode B: 1H EMA20 Pullback ─────────────────────────────────────────────

    def _mode_b(self, cp, cl, op, hi, lo, vol, n, atr14, volma20,
                ema20_1h, n1h, rsi14, rsi14_1h, adx_4h,
                ema50_1h_val=float("nan")) -> tuple[bool, str]:
        """
        1H EMA20 pullback in confirmed trending market.
        Requires 4H trend (checked before call) AND 1H EMA20 > EMA50 alignment.
        This multi-TF alignment prevents entries during trend transitions.
        """
        if n < 5 or n1h < 5:
            return False, ""
        lng = self.direction == "long"

        ema1h = float(ema20_1h[n1h]) if n1h < len(ema20_1h) else float("nan")
        if math.isnan(ema1h):
            return False, ""

        # 1H EMA alignment: EMA20 must confirm same trend direction as 4H
        if not math.isnan(ema50_1h_val):
            if lng  and ema1h <= ema50_1h_val:
                return False, ""
            if not lng and ema1h >= ema50_1h_val:
                return False, ""

        # Genuine EMA bounce: low must pierce EMA20, close must recover near EMA
        # (not just "close to EMA above" — this was causing entries during falling knives)
        if lng:
            if lo[n] > ema1h:                     # low never reached EMA — not a touch
                return False, ""
            if cl[n] < ema1h - 0.5 * atr14:       # fell too far through EMA, not bouncing
                return False, ""
        else:
            if hi[n] < ema1h:                     # high never reached EMA — not a touch
                return False, ""
            if cl[n] > ema1h + 0.5 * atr14:       # rose too far through EMA, not rejecting
                return False, ""

        # 1H RSI zone: longs need 1H in pullback (35-55), shorts need 1H elevated (55-72)
        if not math.isnan(rsi14_1h):
            if lng and not (35 <= rsi14_1h <= 55):
                return False, ""
            if not lng and not (55 <= rsi14_1h <= 72):
                return False, ""

        # 15m RSI must be below midpoint — genuine pullback momentum, not already recovered
        if not math.isnan(rsi14):
            if lng and rsi14 > 50:
                return False, ""
            if not lng and rsi14 < 50:
                return False, ""

        # Confirming bar in trend direction
        body = abs(cl[n] - op[n])
        if lng:
            bar_ok = cl[n] > op[n] and body >= atr14 * 0.2
        else:
            bar_ok = cl[n] < op[n] and body >= atr14 * 0.2
        if not bar_ok:
            return False, ""

        # Volume at least average
        if not math.isnan(volma20) and volma20 > 0 and vol[n] < volma20 * 0.8:
            return False, ""

        vol_r = vol[n] / volma20 if (not math.isnan(volma20) and volma20 > 0) else 1.0
        arrow = "↑" if lng else "↓"
        return True, (f"1H-EMA{arrow} ema={ema1h:.0f} "
                      f"rsi1h={rsi14_1h:.0f} vol×{vol_r:.1f}")

    # ── Mode C: Volume Breakout ───────────────────────────────────────────────

    def _mode_c(self, cp, cl, hi, lo, op, vol, n, atr14, volma20,
                rsi14, ema20_15m) -> tuple[bool, str]:
        """
        Volume breakout of 20-bar range — body-confirmed, not just wick.
        LONG : vol ≥ 2×MA + CLOSE above 20-bar high + body ≥ 0.5×ATR + RSI 45-75
        SHORT: vol ≥ 2×MA + CLOSE below 20-bar low  + body ≥ 0.5×ATR + RSI 25-55

        Body requirement prevents wick-spike false breakouts.
        """
        if n < 22 or math.isnan(volma20) or volma20 <= 0:
            return False, ""
        lng = self.direction == "long"

        # Strong volume (2×MA)
        vol_r = vol[n] / volma20
        if vol_r < 2.0:
            return False, ""

        # Bar body must be ≥ 0.5×ATR (body-confirmed breakout, not wick)
        body = abs(cl[n] - op[n])
        if body < atr14 * 0.5:
            return False, ""

        # Breakout with close confirmation (close fully beyond range boundary)
        if lng:
            range_high = max(hi[n - 20: n])
            if not (cl[n] > range_high and cl[n] > op[n]):   # must close bullish
                return False, ""
            rsi_ok = 45 <= rsi14 <= 75 if not math.isnan(rsi14) else True
        else:
            range_low = min(lo[n - 20: n])
            if not (cl[n] < range_low and cl[n] < op[n]):    # must close bearish
                return False, ""
            rsi_ok = 25 <= rsi14 <= 55 if not math.isnan(rsi14) else True

        if not rsi_ok:
            return False, ""

        # Not overextended from 15m EMA20 (> 3×ATR = chasing)
        if not math.isnan(ema20_15m):
            if abs(cp - ema20_15m) > 3.0 * atr14:
                return False, ""

        arrow = "↑" if lng else "↓"
        return True, f"Break{arrow} vol×{vol_r:.1f} body={body:.0f} rsi={rsi14:.0f}"

    # ── Mode D: RSI Divergence + 4H Structural S/R ───────────────────────────

    def _mode_d(self, cp, cl, op, hi, lo, vol, n, atr14, volma20,
                rsi14, rsi14_arr, rsi14_1h,
                hi4h, lo4h, n4h, atr14_4h, has_4h,
                trend_4h: str = "range") -> tuple[bool, str]:
        """
        15m RSI divergence at a 4H structural level.
        - Bullish div (lower low price, higher low RSI) at 4H swing support → long
        - Bearish div (higher high price, lower high RSI) at 4H swing resistance → short
        High-WR because momentum shifts at institutional S/R levels.
        Shorts require confirmed 4H bear trend (not just range) to avoid false entries
        during market recoveries.
        """
        if n < 20:
            return False, ""
        lng = self.direction == "long"

        # Shorts require confirmed 4H downtrend — avoid shorting during recoveries
        if not lng and trend_4h != "bear":
            return False, ""

        # RSI divergence on 15m (uses last 16 bars)
        div = _rsi_divergence(cl, list(rsi14_arr))
        if lng and div != "bullish":
            return False, ""
        if not lng and div != "bearish":
            return False, ""

        # Price must be near 4H structural S/R level
        if not has_4h or n4h < 20 or len(lo4h) < 20 or len(hi4h) < 20:
            return False, ""
        if lng and not _near_support(cp, lo4h, n4h, atr14_4h):
            return False, ""
        if not lng and not _near_resistance(cp, hi4h, n4h, atr14_4h):
            return False, ""

        # 1H RSI: longs need pullback zone, shorts need elevated zone
        if not math.isnan(rsi14_1h):
            if lng and not (28 <= rsi14_1h <= 62):
                return False, ""
            if not lng and not (38 <= rsi14_1h <= 72):
                return False, ""

        # 15m RSI: confirm oversold/overbought condition
        if not math.isnan(rsi14):
            if lng and rsi14 > 58:
                return False, ""
            if not lng and rsi14 < 42:
                return False, ""

        # Confirming bar in trend direction
        body = abs(cl[n] - op[n])
        if lng:
            bar_ok = cl[n] >= op[n] and body >= atr14 * 0.15
        else:
            bar_ok = cl[n] <= op[n] and body >= atr14 * 0.15
        if not bar_ok:
            return False, ""

        # Volume at least 70% average
        if not math.isnan(volma20) and volma20 > 0 and vol[n] < volma20 * 0.7:
            return False, ""

        arrow = "↑" if lng else "↓"
        return True, (f"RsiDiv{arrow}@4H-SR rsi={rsi14:.0f} rsi1h={rsi14_1h:.0f}")

    # ── Layer 1: Reversal Score (7 conditions) ────────────────────────────────

    def _layer1(self, cp, cl, hi, lo, vol, n,
                rsi14, rsi14_arr, hma20, adx_arr,
                ema50, atr14, volma20) -> tuple[int, list]:
        score  = 0
        passed = []
        lng    = self.direction == "long"

        # [1] RSI threshold — wider than V1 (38 / 62) for more signals
        if lng and not math.isnan(rsi14) and rsi14 < 38:
            score += 1; passed.append(f"RSI<38({rsi14:.0f})")
        elif not lng and not math.isnan(rsi14) and rsi14 > 62:
            score += 1; passed.append(f"RSI>62({rsi14:.0f})")

        # [2] RSI divergence
        div = _rsi_divergence(cl, rsi14_arr)
        if lng and div == "bullish":
            score += 1; passed.append("BullDiv")
        elif not lng and div == "bearish":
            score += 1; passed.append("BearDiv")

        # [3] HMA20 slope supports direction
        slope = _hma_slope(hma20, n)
        if lng and slope >= 0:
            score += 1; passed.append("HMAflat+")
        elif not lng and slope <= 0:
            score += 1; passed.append("HMAflat-")

        # [4] ADX rollover (trend momentum fading)
        if _adx_rollover(adx_arr, n):
            score += 1; passed.append("ADXroll")

        # [5] Volume spike ≥ 1.4×MA (slightly lower than V1's 1.5)
        if not math.isnan(volma20) and volma20 > 0 and vol[n] > volma20 * 1.4:
            score += 1; passed.append(f"VolSpike×{vol[n]/volma20:.1f}")

        # [6] Liquidity sweep (stop-hunt + recovery)
        if lng and _liquidity_sweep_low(hi, lo, cl, n):
            score += 1; passed.append("SweepLow")
        elif not lng and _liquidity_sweep_high(hi, lo, cl, n):
            score += 1; passed.append("SweepHigh")

        # [7] Extension beyond EMA50
        if not math.isnan(ema50):
            dist = (ema50 - cp) if lng else (cp - ema50)
            if dist > 1.5 * atr14:
                score += 1; passed.append(f"Ext+{dist/atr14:.1f}ATR")

        return score, passed

    # ── Layer 2: Context Filter (6 conditions) ────────────────────────────────

    def _layer2(self, cp, rsi14_1h, adx_4h, atr14_4h,
                hi4h, lo4h, n4h, has_4h,
                mtf_bias, ema20_1h, n1h, has_1h,
                atr14_1h, volma_1h) -> tuple[int, list]:
        score  = 0
        passed = []
        lng    = self.direction == "long"

        # [1] 4H ADX < 40 (not in extreme trend)
        if math.isnan(adx_4h) or adx_4h < 40:
            score += 1; passed.append(f"4HADX={adx_4h:.0f}<40" if not math.isnan(adx_4h) else "4HADX?")

        # [2] 1H RSI gating (not already at same extreme)
        if lng:
            if math.isnan(rsi14_1h) or rsi14_1h > 25:
                score += 1; passed.append(f"1HRSI={rsi14_1h:.0f}>25" if not math.isnan(rsi14_1h) else "1HRSI?")
        else:
            if math.isnan(rsi14_1h) or rsi14_1h < 75:
                score += 1; passed.append(f"1HRSI={rsi14_1h:.0f}<75" if not math.isnan(rsi14_1h) else "1HRSI?")

        # [3] 1H volume not collapsing
        if has_1h and n1h >= 3:
            if _vol_not_declining(volma_1h, n1h):
                score += 1; passed.append("VolOK")
        else:
            score += 1; passed.append("VolOK?")

        # [4] Near 4H S/R zone
        if has_4h and len(hi4h) > 20 and len(lo4h) > 20:
            if lng and _near_support(cp, lo4h, n4h, atr14_4h):
                score += 1; passed.append("NearSupp")
            elif not lng and _near_resistance(cp, hi4h, n4h, atr14_4h):
                score += 1; passed.append("NearRes")
        else:
            score += 1; passed.append("SR?")

        # [5] MTF Bias within limit
        if lng and mtf_bias > -self.mtf_bias_limit:
            score += 1; passed.append(f"Bias{mtf_bias:+.0f}")
        elif not lng and mtf_bias < self.mtf_bias_limit:
            score += 1; passed.append(f"Bias{mtf_bias:+.0f}")

        # [6] EMA20 1H slope not fighting direction strongly
        if has_1h and n1h >= 4:
            slope_1h     = _ema20_slope(ema20_1h, n1h)
            slope_per_atr = slope_1h / max(atr14_1h, 1e-10)
            ok = (slope_per_atr > -0.20) if lng else (slope_per_atr < 0.20)
            if ok:
                score += 1; passed.append("EMAslope")
        else:
            score += 1; passed.append("EMAslope?")

        # [7] ATR regime: 1H/4H ATR ratio in healthy range (not over-volatile vs structure)
        if not math.isnan(atr14_1h) and not math.isnan(atr14_4h) and atr14_4h > 0:
            ratio = atr14_1h / atr14_4h
            if 0.08 <= ratio <= 0.60:
                score += 1; passed.append(f"ATRreg={ratio:.2f}")
        else:
            score += 1; passed.append("ATRreg?")

        return score, passed

    # ── Layer 3: Price Action Trigger ─────────────────────────────────────────

    def _layer3(self, cp, cl, hi, lo, op, n, atr14, volma20,
                vol) -> tuple[bool, str, int]:
        lng = self.direction == "long"

        if lng and _choch_bull(hi, cl, n):
            return True, "CHOCH_bull", 1
        if not lng and _choch_bear(lo, cl, n):
            return True, "CHOCH_bear", 1
        if lng and _bos_bull(hi, cl, n):
            return True, "BOS_bull", 1
        if not lng and _bos_bear(lo, cl, n):
            return True, "BOS_bear", 1

        vol_spike = (not math.isnan(volma20) and volma20 > 0
                     and vol[n] > volma20 * 1.5)
        if lng and _double_bottom(lo, cl, n) and vol_spike:
            return True, "DblBottom", 2
        if not lng and _double_top(hi, cl, n) and vol_spike:
            return True, "DblTop", 2

        if lng and _hammer(op[n], hi[n], lo[n], cl[n], atr14):
            return True, "Hammer", 3
        if not lng and _shooting_star(op[n], hi[n], lo[n], cl[n], atr14):
            return True, "ShootStar", 3
        if lng and _bull_engulf(op, cl, n):
            return True, "BullEngulf", 3
        if not lng and _bear_engulf(op, cl, n):
            return True, "BearEngulf", 3
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
        """
        Simplified exit logic — let SL/TP do the heavy lifting.
        Only exit early on extreme events to avoid cutting valid trades short.
        Removing CHOCH / RSI / divergence / candle-pressure early exits was the
        core fix: those fired at -0.5 ATR losses on trades that would have
        recovered to TP, artificially suppressing WR to ~39%.
        """
        lng = self.direction == "long"

        # [1] Health emergency — always fires
        if health_score < 25:
            return f"health={health_score}<25"

        # Minimum hold: give trade 3 bars (45 min) to develop
        if self._bars_in_trade < 3:
            return None

        # [2] RSI recovery exit — primary profit-taking mechanism.
        # Entering when RSI extreme (<38 long / >62 short) then holding until
        # RSI reaches recovery zone (>65 long / <35 short). This captures the
        # mean-reversion move while ensuring meaningful price movement occurred.
        if not self._tp1_hit:
            if not math.isnan(rsi14):
                if lng and rsi14 > 65:
                    return f"RSIRecov>{rsi14:.0f}"
                if not lng and rsi14 < 35:
                    return f"RSIRecov<{rsi14:.0f}"

        # [3] Extreme opposing bar (body ≥ 1.5×ATR + vol ≥ 2×MA).
        # Very rare — signals a genuine momentum flip, not noise.
        if n >= 2 and atr14 > 0 and not math.isnan(volma20) and volma20 > 0:
            body  = abs(cl[n] - op[n])
            vol_r = vol[n] / volma20
            if lng and cl[n] < op[n] and body >= atr14 * 1.5 and vol_r >= 2.0:
                return f"ExtremeMom↓ body={body:.0f}>1.5ATR vol×{vol_r:.1f}"
            if not lng and cl[n] > op[n] and body >= atr14 * 1.5 and vol_r >= 2.0:
                return f"ExtremeMom↑ body={body:.0f}>1.5ATR vol×{vol_r:.1f}"

        # [3] Post-TP1: tighten exits once we've banked some profit
        if self._tp1_hit:
            if lng and _choch_bear(lo, cl, n):
                return "BearCHOCH(post-TP1)"
            if not lng and _choch_bull(hi, cl, n):
                return "BullCHOCH(post-TP1)"
            if not math.isnan(rsi14):
                if lng and rsi14 > 78:
                    return f"RSI>{rsi14:.0f}(post-TP1)"
                if not lng and rsi14 < 22:
                    return f"RSI<{rsi14:.0f}(post-TP1)"
            # Breakeven stop: protect TP1 profit if price retreats to entry zone
            if self._entry_price > 0 and atr14 > 0:
                be_buf = 0.3 * atr14
                if lng and cp <= self._entry_price + be_buf:
                    return f"BEStop(entry={self._entry_price:.2f})"
                if not lng and cp >= self._entry_price - be_buf:
                    return f"BEStop(entry={self._entry_price:.2f})"

        # [4] Time exit — cut stale trade with no progress
        if not self._tp1_hit and self._bars_in_trade >= self.max_bars:
            progress = (cp - self._entry_price) if lng else (self._entry_price - cp)
            if progress < 0.2 * atr14:
                return (f"TimeExit {self._bars_in_trade}bars "
                        f"progress={progress:.0f}")

        return None

    # ── SL / TP ───────────────────────────────────────────────────────────────

    def _calc_sltp(self, cp, hi, lo, n,
                   atr14) -> tuple[float, float, float, float]:
        """
        SL  = max(1.5×ATR14, pattern extreme + 0.3×ATR buffer)
        TP1 = 1.5× SL dist  (internal reference — tighten exits)
        TP  = 3.0× SL dist  (exchange OCO target, R:R = 1:2)
        """
        lng     = self.direction == "long"
        lookback = min(5, n)

        if lng:
            pattern_extreme = min(lo[n - lookback: n + 1])
            sl_pattern = (cp - pattern_extreme) + 0.3 * atr14
            sl_atr     = self.sl_atr_mult * atr14   # 1.5×
            sl_dist    = max(sl_pattern, sl_atr)
            sl_p       = round(cp - sl_dist, 4)
            tp1        = round(cp + 1.5 * sl_dist, 4)
            tp_p       = round(cp + self.tp_mult * sl_dist, 4)
        else:
            pattern_extreme = max(hi[n - lookback: n + 1])
            sl_pattern = (pattern_extreme - cp) + 0.3 * atr14
            sl_atr     = self.sl_atr_mult * atr14
            sl_dist    = max(sl_pattern, sl_atr)
            sl_p       = round(cp + sl_dist, 4)
            tp1        = round(cp - 1.5 * sl_dist, 4)
            tp_p       = round(cp - self.tp_mult * sl_dist, 4)

        return sl_p, tp1, tp_p, sl_dist
