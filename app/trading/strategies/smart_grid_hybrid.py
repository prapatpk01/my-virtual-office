"""
SmartGridHybrid V1 — Adaptive mode strategy: Grid / Swing / Breakout.

Symbols  : BTC/USDT, XAU/USDT
Entry TF : 30m  (from mtf_candles['30m'])
Regime TF: 4H   (from mtf_candles['4h'])
Macro TF : 1D   (from mtf_candles['1d'])
Tick     : 60 s

Market Regime Engine (4H ADX):
  GRID      ADX<18 AND |EMA20-EMA50|/EMA50<1% AND ATR14<ATR50
  SWING     18 <= ADX < 28
  BREAKOUT  ADX >= 28
  Priority  Breakout > Swing > Grid

Grid Mode  (sideway market):
  Step = 0.8×ATR14(30m), max 5 levels, kill when ADX>22 or EMA cross
Swing Mode (5 layers + health ≥ 70):
  L1 EMA20(4H)>EMA50(4H) | L2 ADX>18+rising | L3 pullback ≤ 1×ATR(30m) from EMA20
  L4 5/6 conditions | L5 health ≥ 70
Breakout Mode (6 layers):
  L1 ADX≥28 | L2 Close>HH20 | L3 Vol>1.5×VolMA | L4 MACD>0 | L5 ROC9>0 | L6 RSI<78

Risk: SL=1.8×ATR(30m), TP primary=2R; TP1 1R(30%), TP2 2R(30%), TP3 trail(40%)
Global exit: EMA20(4H)<EMA50(4H) OR Close(4H)<EMA50(4H)×2 OR health<50
"""
import math
from .base import BaseStrategy, Signal, SignalType

_WARMUP_30M = 60
_WARMUP_4H  = 50


def _roc(closes: list, period: int) -> list:
    n = len(closes)
    out = [float('nan')] * n
    for i in range(period, n):
        prev = closes[i - period]
        if abs(prev) > 1e-10:
            out[i] = (closes[i] - prev) / prev * 100.0
    return out


class SmartGridHybrid(BaseStrategy):
    """Regime-adaptive strategy: Grid (sideway) / Swing (trend) / Breakout (strong trend)."""

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.sl_atr        = float(self.params.get("sl_atr",         1.8))
        self.health_thresh = float(self.params.get("health_thresh",  70.0))
        self.adx_swing     = float(self.params.get("adx_swing",     18.0))
        self.adx_breakout  = float(self.params.get("adx_breakout",  28.0))
        self.adx_grid_kill = float(self.params.get("adx_grid_kill", 22.0))
        self.max_grid_lvls = int(self.params.get("max_grid_levels",   5))
        self.grid_step_atr = float(self.params.get("grid_step_atr",  0.8))
        self.grid_slot     = int(self.params.get("grid_slot",        -1))   # -1=auto, 0/1/2=fixed slot

        # In slot mode, give this instance a unique name (L1/L2/L3)
        if self.grid_slot >= 0:
            self.name = f"{self.__class__.__name__}_L{self.grid_slot + 1}"

        # Grid state — reset when leaving grid mode
        self._grid_anchor: float   = None
        self._grid_step:   float   = None
        self._grid_last_buy: float = None   # price of last grid buy

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        mtf  = mtf_candles or {}
        cp   = current_price

        # 30m is the entry TF (falls back to primary candles if not provided)
        c30m = mtf.get("30m") or candles
        n    = len(c30m) - 1
        if n < _WARMUP_30M:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Warmup 30m ({n}/{_WARMUP_30M})")

        closes_30m  = [float(c.close)  for c in c30m]
        lows_30m    = [float(c.low)    for c in c30m]
        highs_30m   = [float(c.high)   for c in c30m]
        volumes_30m = [float(c.volume) for c in c30m]

        # ── 4H regime data ────────────────────────────────────────────────
        c4h = mtf.get("4h", [])
        if len(c4h) < _WARMUP_4H:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Need 4H data ({len(c4h)} bars)")

        closes_4h = [float(c.close) for c in c4h]
        m = len(c4h) - 1

        ema20_4h = self.ema(closes_4h, 20)
        ema50_4h = self.ema(closes_4h, 50)
        e20_4h   = float(ema20_4h[m])
        e50_4h   = float(ema50_4h[m])

        adx_arr, _, _ = self.adx(c4h, 14)
        adx_v4   = float(adx_arr[m])
        adx_prev = float(adx_arr[m - 1]) if m >= 1 else adx_v4

        atr14_4h_arr = self.atr(c4h, 14)
        atr50_4h_arr = self.atr(c4h, 50) if len(c4h) >= 55 else atr14_4h_arr
        atr14_4h = float(atr14_4h_arr[m])
        atr50_4h = float(atr50_4h_arr[m])

        # ── 1D macro data ─────────────────────────────────────────────────
        c1d = mtf.get("1d", [])
        has_1d = len(c1d) >= 55
        if has_1d:
            closes_1d = [float(c.close) for c in c1d]
            d = len(c1d) - 1
            e50_1d   = float(self.ema(closes_1d, 50)[d])
            rsi_1d   = float(self.rsi(closes_1d, 14)[d])
            ema200_d = self.ema(closes_1d, 200) if len(c1d) >= 200 else None
            e200_1d  = float(ema200_d[d]) if ema200_d is not None else float('nan')
        else:
            e50_1d = e200_1d = rsi_1d = float('nan')

        # ── 30m ATR ───────────────────────────────────────────────────────
        atr14_arr = self.atr(c30m, 14)
        atr50_arr = self.atr(c30m, 50) if len(c30m) >= 55 else atr14_arr
        atr14_30m = float(atr14_arr[n])
        atr50_30m = float(atr50_arr[n])
        if math.isnan(atr14_30m):
            atr14_30m = cp * 0.005
        if math.isnan(atr50_30m):
            atr50_30m = atr14_30m

        # ── Health score (used for global exit and swing L5) ──────────────
        volma_30m = self.sma(volumes_30m, 20)
        health = self._health(e20_4h, e50_4h, adx_v4, adx_prev,
                              e50_1d, e200_1d, rsi_1d,
                              closes_30m, lows_30m, volumes_30m, volma_30m, n)

        # ── Global exit ───────────────────────────────────────────────────
        death_cross  = (not (math.isnan(e20_4h) or math.isnan(e50_4h))
                        and e20_4h < e50_4h)
        c_below_e50  = not math.isnan(e50_4h) and closes_4h[m] < e50_4h
        p_below_e50  = (m >= 1 and not math.isnan(float(ema50_4h[m - 1]))
                        and closes_4h[m - 1] < float(ema50_4h[m - 1]))
        health_fail  = health < 50

        if death_cross or (c_below_e50 and p_below_e50) or health_fail:
            trigger = ("EMA20<EMA50(4H)" if death_cross else
                       "Close<EMA50(4H)×2" if (c_below_e50 and p_below_e50) else
                       f"Health={health}<50")
            self._reset_grid()
            return Signal(SignalType.SELL, self.symbol, cp, 0,
                          f"[{self.name}] Global exit: {trigger}",
                          metadata={"trigger": trigger})

        # ── Regime ────────────────────────────────────────────────────────
        regime = self._regime(adx_v4, e20_4h, e50_4h, atr14_4h, atr50_4h)

        if regime == "breakout":
            self._reset_grid()
            return self._breakout(closes_30m, highs_30m, volumes_30m, n,
                                  atr14_30m, adx_v4, cp)

        if regime == "swing":
            self._reset_grid()
            return self._swing(closes_30m, lows_30m, volumes_30m, n,
                               atr14_30m, adx_v4, adx_prev, e20_4h, e50_4h,
                               health, cp)

        if regime == "grid":
            return self._grid(closes_30m, highs_30m, lows_30m, volumes_30m, n,
                              atr14_30m, atr50_30m, adx_v4, e20_4h, e50_4h, cp)

        self._reset_grid()
        return Signal(SignalType.HOLD, self.symbol, cp, 0,
                      f"[{self.name}] No regime (ADX={adx_v4:.1f})")

    # ─────────────────────────────────────────────────────────────────────
    # Regime Engine
    # ─────────────────────────────────────────────────────────────────────

    def _regime(self, adx: float, e20: float, e50: float,
                atr14: float, atr50: float) -> str:
        if math.isnan(adx):
            return "none"
        if adx >= self.adx_breakout:
            return "breakout"
        if adx >= self.adx_swing:
            return "swing"
        if math.isnan(e20) or math.isnan(e50) or e50 == 0:
            return "none"
        ema_diff_pct = abs(e20 - e50) / e50
        atr_ok = math.isnan(atr50) or atr14 < atr50
        if ema_diff_pct < 0.01 and atr_ok:
            return "grid"
        return "none"

    # ─────────────────────────────────────────────────────────────────────
    # GRID MODE
    # ─────────────────────────────────────────────────────────────────────

    def _grid(self, closes: list, highs: list, lows: list, volumes: list, n: int,
              atr14: float, atr50: float, adx: float,
              e20_4h: float, e50_4h: float, cp: float) -> Signal:

        # Kill switch: ADX rising above threshold OR EMA cross
        ema_cross = (not (math.isnan(e20_4h) or math.isnan(e50_4h))
                     and abs(e20_4h - e50_4h) / max(abs(e50_4h), 1e-10) > 0.015)
        if adx > self.adx_grid_kill or ema_cross:
            self._reset_grid()
            return Signal(SignalType.SELL, self.symbol, cp, 0,
                          f"[{self.name}] Grid kill (ADX={adx:.1f} emaX={ema_cross})",
                          metadata={"mode": "grid", "trigger": "kill_switch"})

        # Grid entry conditions: RSI 40-60, ATR not spiking
        rsi14 = self.rsi(closes, 14)
        rsi_v = float(rsi14[n])
        rsi_ok = math.isnan(rsi_v) or (40.0 <= rsi_v <= 60.0)
        atr_ok = atr14 <= atr50 * 1.25

        if not (rsi_ok and atr_ok):
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Grid HOLD RSI={rsi_v:.1f} ATR={'ok' if atr_ok else 'spike'}")

        # Initialise grid
        if self._grid_anchor is None or self._grid_step is None:
            self._grid_anchor   = cp
            self._grid_step     = self.grid_step_atr * atr14
            self._grid_last_buy = None

        step   = max(self._grid_step, atr14 * 0.3)  # never smaller than 30% ATR
        anchor = self._grid_anchor

        # Slot mode: each instance targets exactly one fixed price level
        if self.grid_slot >= 0:
            target = anchor - (self.grid_slot + 1) * step
            if self._grid_last_buy is not None:
                tp_ref = self._grid_last_buy + step
                sl_ref = self._grid_last_buy - self.sl_atr * atr14
                if cp >= tp_ref * 0.995 or cp <= sl_ref * 1.005:
                    self._grid_last_buy = None
            if abs(cp - target) > step * 0.25:
                return Signal(SignalType.HOLD, self.symbol, cp, 0,
                              f"[{self.name}] Grid: price not at L{self.grid_slot+1}={target:.2f}")
            if self._grid_last_buy is not None:
                return Signal(SignalType.HOLD, self.symbol, cp, 0,
                              f"[{self.name}] Grid: L{self.grid_slot+1} already in position")
            tp_p = round(target + step, 4)
            sl_p = round(target - self.sl_atr * atr14, 4)
            self._grid_last_buy = target
            return Signal(
                type=SignalType.BUY,
                symbol=self.symbol,
                price=cp,
                amount=0.0,
                confidence=0.62,
                reason=(f"[{self.name}] Grid L{self.grid_slot+1} "
                        f"target={target:.2f} step={step:.2f} RSI={rsi_v:.1f}"),
                metadata={
                    "mode":        "grid",
                    "stop_loss":   sl_p,
                    "take_profit": tp_p,
                    "atr":         round(atr14, 4),
                    "grid_anchor": round(anchor, 4),
                    "grid_step":   round(step, 4),
                    "grid_level":  round(target, 4),
                    "grid_slot":   self.grid_slot,
                    "rsi":         round(rsi_v, 1) if not math.isnan(rsi_v) else None,
                },
            )

        # Infer if previous bot position was closed (price moved back up past TP)
        if self._grid_last_buy is not None:
            tp_ref = self._grid_last_buy + step
            sl_ref = self._grid_last_buy - self.sl_atr * atr14
            if cp >= tp_ref * 0.995:   # TP hit — advance anchor up
                self._grid_anchor = tp_ref
                anchor = tp_ref
                self._grid_last_buy = None
            elif cp <= sl_ref * 1.005:  # SL hit — step anchor down
                self._grid_anchor = self._grid_last_buy - step
                anchor = self._grid_anchor
                self._grid_last_buy = None

        # Compute grid levels below anchor
        levels = [anchor - i * step for i in range(1, self.max_grid_lvls + 1)]

        # Find the grid level closest to the current price (within ±25% of step)
        target = None
        for lvl in levels:
            if abs(cp - lvl) <= step * 0.25:
                target = lvl
                break

        if target is None:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Grid HOLD: price not near any level "
                          f"(anchor={anchor:.2f} step={step:.2f} RSI={rsi_v:.1f})")

        # Avoid buying the same level twice before it clears
        if (self._grid_last_buy is not None
                and abs(target - self._grid_last_buy) < step * 0.4):
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Grid HOLD: level {target:.2f} already bought")

        tp_p = round(target + step, 4)
        sl_p = round(target - self.sl_atr * atr14, 4)
        self._grid_last_buy = target

        return Signal(
            type=SignalType.BUY,
            symbol=self.symbol,
            price=cp,
            amount=0.0,
            confidence=0.62,
            reason=f"[{self.name}] Grid buy lvl={target:.2f} step={step:.2f} RSI={rsi_v:.1f}",
            metadata={
                "mode":        "grid",
                "stop_loss":   sl_p,
                "take_profit": tp_p,
                "atr":         round(atr14, 4),
                "grid_anchor": round(anchor, 4),
                "grid_step":   round(step, 4),
                "grid_level":  round(target, 4),
                "rsi":         round(rsi_v, 1) if not math.isnan(rsi_v) else None,
            },
        )

    # ─────────────────────────────────────────────────────────────────────
    # SWING MODE
    # ─────────────────────────────────────────────────────────────────────

    def _swing(self, closes: list, lows: list, volumes: list, n: int,
               atr14: float, adx: float, adx_prev: float,
               e20_4h: float, e50_4h: float, health: int, cp: float) -> Signal:

        # L1: 4H trend
        if math.isnan(e20_4h) or math.isnan(e50_4h) or e20_4h <= e50_4h:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Swing L1: EMA20(4H)<=EMA50(4H)")

        # L2: ADX > 18 and rising
        if math.isnan(adx) or adx < self.adx_swing:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Swing L2: ADX={adx:.1f} < {self.adx_swing}")
        if adx < adx_prev:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Swing L2: ADX falling ({adx:.1f})")

        # L3: Pullback — close within 1×ATR of EMA20(30m)
        ema20_30 = self.ema(closes, 20)
        e20_30   = float(ema20_30[n])
        dist     = abs(cp - e20_30) if not math.isnan(e20_30) else float('inf')
        if dist > atr14:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Swing L3: pullback dist={dist:.2f} > ATR={atr14:.2f}")

        # L4: Entry score 5/6
        ema5  = self.ema(closes, 5)
        sma9  = self.sma(closes, 9)
        hma16 = self.hma(closes, 16)
        rsi14 = self.rsi(closes, 14)
        volma = self.sma(volumes, 20)
        _, _, mh = self.macd(closes, 12, 26, 9)
        roc9  = _roc(closes, 9)

        e5    = float(ema5[n]);   s9    = float(sma9[n])
        hma   = float(hma16[n]); rsi_v = float(rsi14[n])
        vma   = float(volma[n]); vol   = volumes[n]
        mhc   = float(mh[n])
        roc9v = float(roc9[n]) if n < len(roc9) else float('nan')

        s = 0
        if not (math.isnan(e5) or math.isnan(s9))   and e5 > s9:       s += 1
        if not math.isnan(hma)                        and closes[n] > hma: s += 1
        if not math.isnan(rsi_v)                      and rsi_v > 50.0:   s += 1
        if not math.isnan(roc9v)                      and roc9v > 0.0:    s += 1
        if not math.isnan(mhc)                        and mhc > 0.0:      s += 1
        if not math.isnan(vma) and vma > 0            and vol > vma:      s += 1

        if s < 5:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Swing L4: {s}/6 (need 5)")

        # L5: Health score
        if health < self.health_thresh:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] Swing L5: Health={health} < {self.health_thresh:.0f}")

        sl_dist = self.sl_atr * atr14
        sl_p    = round(cp - sl_dist, 4)
        tp2     = round(cp + 2 * sl_dist, 4)  # 2R — primary OCO TP
        tp1     = round(cp + sl_dist, 4)       # 1R reference

        return Signal(
            type=SignalType.BUY,
            symbol=self.symbol,
            price=cp,
            amount=0.0,
            confidence=min(0.65 + s * 0.04, 0.90),
            reason=f"[{self.name}] Swing L4={s}/6 Health={health} ADX={adx:.1f}",
            metadata={
                "mode":        "swing",
                "stop_loss":   sl_p,
                "take_profit": tp2,
                "tp1_1R":      tp1,
                "atr":         round(atr14, 4),
                "sl_atr":      self.sl_atr,
                "score_l4":    s,
                "health":      health,
                "adx_4h":      round(adx, 1),
                "rsi":         round(rsi_v, 1) if not math.isnan(rsi_v) else None,
            },
        )

    # ─────────────────────────────────────────────────────────────────────
    # BREAKOUT MODE
    # ─────────────────────────────────────────────────────────────────────

    def _breakout(self, closes: list, highs: list, volumes: list, n: int,
                  atr14: float, adx: float, cp: float) -> Signal:

        # L2: Close > Highest High of last 20 bars
        hh20 = max(highs[max(0, n - 19):n + 1])
        if closes[n] <= hh20:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] BO L2: Close {closes[n]:.2f} <= HH20 {hh20:.2f}")

        # L3: Volume > 1.5× VolMA20
        volma = self.sma(volumes, 20)
        vma   = float(volma[n])
        if not math.isnan(vma) and vma > 0 and volumes[n] < vma * 1.5:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] BO L3: volume < 1.5×VolMA")

        # L4: MACD hist > 0
        _, _, mh = self.macd(closes, 12, 26, 9)
        mhc = float(mh[n])
        if not math.isnan(mhc) and mhc <= 0:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] BO L4: MACD hist ≤ 0")

        # L5: ROC9 > 0
        roc9  = _roc(closes, 9)
        roc9v = float(roc9[n]) if n < len(roc9) else float('nan')
        if not math.isnan(roc9v) and roc9v <= 0:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] BO L5: ROC9 ≤ 0")

        # L6: RSI14 < 78
        rsi14 = self.rsi(closes, 14)
        rsi_v = float(rsi14[n])
        if not math.isnan(rsi_v) and rsi_v >= 78.0:
            return Signal(SignalType.HOLD, self.symbol, cp, 0,
                          f"[{self.name}] BO L6: RSI={rsi_v:.1f} ≥ 78 (overbought)")

        sl_dist = self.sl_atr * atr14
        sl_p    = round(cp - sl_dist, 4)
        tp2     = round(cp + 2 * sl_dist, 4)
        tp1     = round(cp + sl_dist, 4)

        return Signal(
            type=SignalType.BUY,
            symbol=self.symbol,
            price=cp,
            amount=0.0,
            confidence=0.82,
            reason=f"[{self.name}] Breakout HH20={hh20:.2f} ADX={adx:.1f}",
            metadata={
                "mode":        "breakout",
                "stop_loss":   sl_p,
                "take_profit": tp2,
                "tp1_1R":      tp1,
                "atr":         round(atr14, 4),
                "sl_atr":      self.sl_atr,
                "hh20":        round(hh20, 4),
                "adx_4h":      round(adx, 1),
                "rsi":         round(rsi_v, 1) if not math.isnan(rsi_v) else None,
            },
        )

    # ─────────────────────────────────────────────────────────────────────
    # Health Score (max 100, need ≥ 70 for swing; global exit < 50)
    # ─────────────────────────────────────────────────────────────────────

    def _health(self, e20_4h: float, e50_4h: float, adx: float, adx_prev: float,
                e50_1d: float, e200_1d: float, rsi_1d: float,
                closes: list, lows: list, volumes: list,
                volma_arr, n: int) -> int:
        h = 0

        # 4H Trend (20 pts)
        if not (math.isnan(e20_4h) or math.isnan(e50_4h)) and e20_4h > e50_4h:
            h += 20

        # ADX Rising (15 pts)
        if not (math.isnan(adx) or math.isnan(adx_prev)) and adx > adx_prev:
            h += 15

        # Momentum (15 pts) — MACD hist > 0 OR ROC9 > 0
        if len(closes) > 30:
            _, _, mh = self.macd(closes, 12, 26, 9)
            mhc   = float(mh[n]) if n < len(mh) else float('nan')
            roc9  = _roc(closes, 9)
            roc9v = float(roc9[n]) if n < len(roc9) else float('nan')
            if (not math.isnan(mhc) and mhc > 0) or (not math.isnan(roc9v) and roc9v > 0):
                h += 15

        # Volume (15 pts)
        vma = float(volma_arr[n]) if n < len(volma_arr) else float('nan')
        if not math.isnan(vma) and vma > 0 and volumes[n] > vma:
            h += 15

        # Structure HL (15 pts) — recent lows higher than prior lows
        if n >= 10:
            recent_low = min(lows[max(0, n - 4):n + 1])
            prior_low  = min(lows[max(0, n - 9):n - 4])
            if recent_low > prior_low:
                h += 15

        # Macro Trend (20 pts) — 1D EMA50>EMA200 OR 1D RSI>50
        if not (math.isnan(e50_1d) or math.isnan(rsi_1d)):
            macro_bull = (not math.isnan(e200_1d) and e50_1d > e200_1d) or rsi_1d > 50.0
            if macro_bull:
                h += 20

        return h

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _reset_grid(self):
        self._grid_anchor   = None
        self._grid_step     = None
        self._grid_last_buy = None
