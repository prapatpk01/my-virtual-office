"""
SJ-MACD/EMA Strategy.

Signal fires on MACD histogram crossover with price above HMA:
  BUY:  MACD histogram crosses from ≤0 to >0 AND ha_close > HMA
  SELL: MACD histogram crosses from ≥0 to <0

Default: MACD 5/13/4, HMA9 — fires ~15-25 signals/250 bars 1H on MACD hist crossover
  ATR 14, Stop 1.5×ATR, R:R 1:1.4
"""
import logging
import numpy as np
from .base import BaseStrategy, Signal, SignalType

logger = logging.getLogger("macd_ema_strategy")

_LOOKFORWARD = 60
_SL_MULTS    = [1.0, 1.5, 2.0, 2.5]


class _HAC:
    __slots__ = ("timestamp", "open", "high", "low", "close", "volume")
    def __init__(self, ts, o, h, l, c, v):
        self.timestamp = ts
        self.open = o; self.high = h; self.low = l; self.close = c; self.volume = v


class MACDEMAStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.hma_period  = self.params.get("hma_period",   9)   # fast trend filter
        self.ema_fast    = self.params.get("ema_fast",     5)
        self.sma_slow    = self.params.get("sma_slow",    10)
        self.macd_fast   = self.params.get("macd_fast",    5)
        self.macd_slow   = self.params.get("macd_slow",   13)
        self.macd_sig    = self.params.get("macd_signal",  4)
        self.atr_period  = self.params.get("atr_period",  14)
        self.sl_atr_mult = self.params.get("sl_atr_mult", 1.5)
        self.rr_ratio    = self.params.get("rr_ratio",    1.4)

    # ── Heikin Ashi ────────────────────────────────────────────────

    @staticmethod
    def _heikin_ashi(candles: list):
        n = len(candles)
        ha_o = np.zeros(n); ha_c = np.zeros(n)
        ha_h = np.zeros(n); ha_l = np.zeros(n)
        for i, c in enumerate(candles):
            ha_c[i] = (c.open + c.high + c.low + c.close) / 4.0
            ha_o[i] = ((c.open + c.close) / 2.0 if i == 0
                       else (ha_o[i-1] + ha_c[i-1]) / 2.0)
            ha_h[i] = max(c.high, ha_o[i], ha_c[i])
            ha_l[i] = min(c.low,  ha_o[i], ha_c[i])
        ha_candles = [
            _HAC(candles[i].timestamp, ha_o[i], ha_h[i], ha_l[i], ha_c[i], candles[i].volume)
            for i in range(n)
        ]
        return ha_candles, ha_o, ha_c

    # ── Signal logic ───────────────────────────────────────────────

    def _signal_at(self, i: int,
                   ha_o, ha_c, ha_highs, ha_lows,
                   hma, ema, sma, ml, sl_line) -> int:
        """Returns +1 BUY, -1 SELL, 0 HOLD.
        BUY:  MACD histogram crosses ≤0→>0 while ha_close > HMA
        SELL: MACD histogram crosses ≥0→<0
        """
        if i < 1:
            return 0
        if any(np.isnan(v) for v in [hma[i], ml[i], ml[i-1], sl_line[i], sl_line[i-1], ha_c[i]]):
            return 0

        hist_c = float(ml[i])   - float(sl_line[i])
        hist_p = float(ml[i-1]) - float(sl_line[i-1])

        if hist_c > 0 and hist_p <= 0 and float(ha_c[i]) > float(hma[i]):
            return 1
        if hist_c < 0 and hist_p >= 0:
            return -1
        return 0

    # ── Pre-compute arrays ─────────────────────────────────────────

    def _build_arrays(self, candles: list):
        ha_candles, ha_o, ha_c = self._heikin_ashi(candles)

        ha_highs = np.array([c.high for c in ha_candles], dtype=float)
        ha_lows  = np.array([c.low  for c in ha_candles], dtype=float)

        cl = ha_c.tolist()
        hma     = np.array(self.hma(cl, self.hma_period),                              dtype=float)
        ema     = np.array(self.ema(cl, self.ema_fast),                                dtype=float)
        sma     = np.array(self.sma(cl, self.sma_slow),                                dtype=float)
        _ml, _sl, _hi = self.macd(cl, self.macd_fast, self.macd_slow, self.macd_sig)
        ml      = np.array(_ml, dtype=float)
        sl_line = np.array(_sl, dtype=float)
        hist    = np.array(_hi, dtype=float)
        atr_a   = np.array(self.atr(ha_candles, self.atr_period), dtype=float)

        return (ha_o, ha_c, ha_highs, ha_lows,
                hma, ema, sma, ml, sl_line, hist, atr_a)

    # ── Live analysis ──────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = self.macd_slow + self.macd_sig + self.hma_period + 5
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        (ha_o, ha_c, ha_highs, ha_lows,
         hma, ema, sma, ml, sl_line, hist, atr_a) = self._build_arrays(candles)

        n = len(candles) - 1
        direction = self._signal_at(
            n, ha_o, ha_c, ha_highs, ha_lows,
            hma, ema, sma, ml, sl_line,
        )

        p     = current_price
        atr_c = float(atr_a[n]) if not np.isnan(atr_a[n]) else 0.0

        meta = {
            "ha_open":  round(float(ha_o[n]), 4),
            "ha_close": round(float(ha_c[n]), 4),
            "hma20":    round(float(hma[n]),  4),
            "ema":      round(float(ema[n]),  4),
            "sma":      round(float(sma[n]),  4),
            "macd":     round(float(ml[n]),   5),
            "signal":   round(float(sl_line[n]), 5),
            "hist":     round(float(hist[n]), 5),
            "atr":      round(atr_c, 4),
        }

        if direction == 1:
            sl_p = round(p - self.sl_atr_mult * atr_c, 4)
            tp_p = round(p + self.sl_atr_mult * self.rr_ratio * atr_c, 4)
            conf = round(min(0.85, 0.65 + abs(float(ml[n]) - float(sl_line[n])) / max(atr_c, 1) * 5), 2)
            return Signal(
                type=SignalType.BUY, symbol=self.symbol, price=p, amount=0.0,
                confidence=conf,
                reason=f"[MACD/EMA] BUY | open>{float(hma[n]):.4f} EMA×↑ MACD×↑",
                metadata={**meta, "stop_loss": sl_p, "take_profit": tp_p, "rr": self.rr_ratio},
            )

        if direction == -1:
            sl_p = round(p + self.sl_atr_mult * atr_c, 4)
            tp_p = round(p - self.sl_atr_mult * self.rr_ratio * atr_c, 4)
            conf = round(min(0.85, 0.65 + abs(float(ml[n]) - float(sl_line[n])) / max(atr_c, 1) * 5), 2)
            return Signal(
                type=SignalType.SELL, symbol=self.symbol, price=p, amount=0.0,
                confidence=conf,
                reason=f"[MACD/EMA] SELL | open<{float(hma[n]):.4f} EMA×↓ MACD×↓",
                metadata={**meta, "stop_loss": sl_p, "take_profit": tp_p, "rr": self.rr_ratio},
            )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[MACD/EMA] HOLD | EMA={float(ema[n]):.4f} SMA={float(sma[n]):.4f}",
            metadata=meta,
        )

    # ── Backtest ───────────────────────────────────────────────────

    async def backtest(self, candles: list) -> tuple[dict, tuple]:
        min_len = self.macd_slow + self.macd_sig + self.hma_period + 20
        if len(candles) < min_len:
            return {}, None

        (ha_o, ha_c, ha_highs, ha_lows,
         hma, ema, sma, ml, sl_line, hist, atr_a) = self._build_arrays(candles)

        signal_bars: list[tuple[int, int, float]] = []
        prev_dir = 0

        for i in range(min_len, len(candles) - 1):
            d = self._signal_at(i, ha_o, ha_c, ha_highs, ha_lows,
                                hma, ema, sma, ml, sl_line)
            if d == 1 and prev_dir != 1:
                signal_bars.append((i, 1, float(atr_a[i])))
                prev_dir = 1
            elif d == -1 and prev_dir != -1:
                signal_bars.append((i, -1, float(atr_a[i])))
                prev_dir = -1
            elif d == 0:
                prev_dir = 0

        if not signal_bars:
            return {}, None

        best_score, best_config = -999.0, None
        stats: dict = {}

        for sl_m in _SL_MULTS:
            rr      = self.rr_ratio
            wins    = losses = 0
            total_r = 0.0
            for idx, direction, atr_val in signal_bars:
                if atr_val <= 0:
                    continue
                entry = float(ha_c[idx])
                sl_p  = entry - sl_m * atr_val if direction == 1 else entry + sl_m * atr_val
                tp_p  = entry + sl_m * rr * atr_val if direction == 1 else entry - sl_m * rr * atr_val
                outcome = 0
                for j in range(idx + 1, min(idx + _LOOKFORWARD, len(ha_c))):
                    if direction == 1:
                        if ha_lows[j]  <= sl_p: outcome = -1; break
                        if ha_highs[j] >= tp_p: outcome =  1; break
                    else:
                        if ha_highs[j] >= sl_p: outcome = -1; break
                        if ha_lows[j]  <= tp_p: outcome =  1; break
                if outcome ==  1: wins   += 1; total_r += rr
                elif outcome == -1: losses += 1; total_r -= 1.0

            total = wins + losses
            wr    = wins / total if total else 0.0
            pf    = (wins * rr) / max(losses, 1)
            key   = f"SL={sl_m}xATR  RR=1:{rr}"
            stats[key] = {
                "win_rate": round(wr * 100, 1), "profit_factor": round(pf, 2),
                "total_r":  round(total_r, 1),  "trades": total,
                "wins":     wins,                "losses": losses,
            }
            if total >= 5 and total_r > best_score:
                best_score  = total_r
                best_config = (sl_m, rr)

        return stats, best_config
