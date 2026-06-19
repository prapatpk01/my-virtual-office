"""Intern Trader Strategy (เด็กฝึกงาน) — HMA(15) timing + MTF 15m/30m bias filter.

Entry (BUY — Chief must still confirm):
  1. Previous HA candle closed ABOVE HMA(hma_len)
  2. Current  HA candle opens ABOVE HMA(hma_len)
  3. Current  HA open > previous HA open
  4. MTF bias: 15m score ≥ 1  AND  30m score ≥ 1
     (score = EMA20/50 cross + RSI55/45, range −3…+3)

Exit (SELL — no Chief needed, immediate close):
  1. Previous HA candle closed BELOW HMA(hma_len)
  2. Current  HA candle opens  BELOW HMA(hma_len)

Candles passed in are already Heikin Ashi (bot.py converts before calling).
MTF candles (15m / 30m) are fetched internally and also converted to HA.
"""
import math
import numpy as np

from .base import BaseStrategy, Signal, SignalType

_WARMUP = 60   # need at least this many HA candles for stable HMA(15)


def _tf_score(candles: list) -> float:
    """Per-timeframe bias score in [−3, +3].

    +1  close > EMA20          else −1
    +1  EMA20 > EMA50          else −1
    +1  RSI14 > 55             else (−1 if RSI14 < 45 else 0)
    """
    if len(candles) < 52:
        return 0.0
    closes = [float(c.close) for c in candles]
    e20 = float(BaseStrategy.ema(closes, 20)[-1])
    e50 = float(BaseStrategy.ema(closes, 50)[-1])
    rsi = float(BaseStrategy.rsi(closes, 14)[-1])
    if any(math.isnan(v) for v in (e20, e50, rsi)):
        return 0.0
    price = closes[-1]
    return (
        (1.0 if price > e20 else -1.0) +
        (1.0 if e20  > e50 else -1.0) +
        (1.0 if rsi  > 55.0 else (-1.0 if rsi < 45.0 else 0.0))
    )


class InternStrategy(BaseStrategy):
    """HMA(15) + dual-TF bias entry / exit strategy.

    Passed to TradingBot alongside other strategies.
    BUY signals are queued for Chief confirmation.
    SELL signals trigger immediate early exit (no Chief needed).
    """

    _DEFAULTS = dict(hma_len=15, sl_atr=2.5, tp_atr=2.0,
                     mtf_lo="15m", mtf_mid="30m")

    def __init__(self, symbol: str, params: dict = None, connector=None):
        super().__init__(symbol, params)
        d = self._DEFAULTS
        self.hma_len = int(self.params.get("hma_len", d["hma_len"]))
        self.sl_atr  = float(self.params.get("sl_atr",  d["sl_atr"]))
        self.tp_atr  = float(self.params.get("tp_atr",  d["tp_atr"]))
        self.mtf_lo  = str(self.params.get("mtf_lo",   d["mtf_lo"]))
        self.mtf_mid = str(self.params.get("mtf_mid",  d["mtf_mid"]))
        self._connector = connector
        self._cooldown  = 0

    # ── Helpers ────────────────────────────────────────────────────────

    async def _mtf_bullish(self) -> tuple[bool, str]:
        """Fetch 15m and 30m HA candles; return (both_bullish, label)."""
        if self._connector is None:
            return True, "no-connector(skipped)"

        from ..connectors.base import to_heikin_ashi

        scores: dict[str, float] = {}
        for tf in (self.mtf_lo, self.mtf_mid):
            try:
                raw = await self._connector.fetch_ohlcv(
                    self.symbol, timeframe=tf, limit=80
                )
                ha = to_heikin_ashi(raw)
                scores[tf] = _tf_score(ha)
            except Exception:
                scores[tf] = 0.0

        def _arrow(s: float) -> str:
            return "↑↑" if s >= 2 else "↑" if s > 0 else "↓↓" if s <= -2 else "↓" if s < 0 else "→"

        lo_ok  = scores.get(self.mtf_lo,  0.0) >= 1.0
        mid_ok = scores.get(self.mtf_mid, 0.0) >= 1.0
        label  = (
            f"{self.mtf_lo}{_arrow(scores.get(self.mtf_lo, 0))}"
            f" {self.mtf_mid}{_arrow(scores.get(self.mtf_mid, 0))}"
        )
        return (lo_ok and mid_ok), label

    # ── Core ───────────────────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        n = len(candles)
        if n < _WARMUP:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] Warming up ({n}/{_WARMUP})")

        closes = np.array([c.close for c in candles], dtype=float)
        opens  = np.array([c.open  for c in candles], dtype=float)
        hma    = self.hma(closes.tolist(), self.hma_len)

        prev_i = n - 2
        curr_i = n - 1

        hma_prev = float(hma[prev_i])
        hma_curr = float(hma[curr_i])
        if math.isnan(hma_prev) or math.isnan(hma_curr):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] HMA NaN — need more bars")

        prev_close = float(closes[prev_i])
        curr_open  = float(opens[curr_i])
        prev_open  = float(opens[prev_i])

        # ── EXIT: both prev close AND current open below HMA ──────────
        # Checked before cooldown so an exit always fires even mid-cooldown.
        if prev_close < hma_prev and curr_open < hma_curr:
            return Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                price=current_price,
                amount=0.0,
                confidence=0.80,
                reason=(
                    f"[{self.name}] HA prev_close={prev_close:.4f} curr_open={curr_open:.4f}"
                    f" < HMA{self.hma_len}={hma_curr:.4f}"
                ),
            )

        # ── Cooldown guard ────────────────────────────────────────────
        if self._cooldown > 0:
            self._cooldown -= 1
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] Cooldown {self._cooldown} bars left")

        # ── HMA BUY pre-conditions ────────────────────────────────────
        hma_ok = (
            prev_close > hma_prev    # previous HA close above HMA
            and curr_open > hma_curr  # current HA open above HMA
            and curr_open > prev_open  # current HA open > previous HA open
        )
        if not hma_ok:
            return Signal(
                SignalType.HOLD, self.symbol, current_price, 0,
                f"[{self.name}] HMA gate fail: prev_c={prev_close:.4f}"
                f" curr_o={curr_open:.4f} prev_o={prev_open:.4f} HMA={hma_curr:.4f}",
            )

        # ── MTF 15m/30m bias ─────────────────────────────────────────
        mtf_ok, mtf_label = await self._mtf_bullish()
        if not mtf_ok:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[{self.name}] MTF not bullish: {mtf_label}")

        # ── All conditions met → BUY ──────────────────────────────────
        atr_arr = self.atr(candles, 14)
        atr_val = float(atr_arr[curr_i])
        if math.isnan(atr_val):
            atr_val = current_price * 0.015

        sl_p = round(current_price - self.sl_atr * atr_val, 4)
        tp_p = round(current_price + self.tp_atr * atr_val, 4)

        self._cooldown = 3   # don't re-fire until cooldown expires
        return Signal(
            type=SignalType.BUY,
            symbol=self.symbol,
            price=current_price,
            amount=0.0,
            confidence=0.72,
            reason=(
                f"[{self.name}] HMA{self.hma_len}↑"
                f" open({curr_open:.4f})>prevOpen({prev_open:.4f})"
                f" | {mtf_label}"
            ),
            metadata={
                "stop_loss":   sl_p,
                "take_profit": tp_p,
                "atr":         round(atr_val, 4),
                "sl_atr":      self.sl_atr,
                "tp_atr":      self.tp_atr,
                "hma":         round(hma_curr, 4),
                "mtf_label":   mtf_label,
            },
        )
