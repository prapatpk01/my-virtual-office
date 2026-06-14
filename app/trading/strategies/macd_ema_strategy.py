"""
MACD + EMA Strategy — Institutional Momentum (SJ rewrite).

Uses Heikin Ashi candles for all indicator calculations.

All conditions must pass (AND logic):
  1. 1H EMA50 trend  — HA_close > EMA50_1h for BUY
  2. HMA15 slope     — rising for BUY, falling for SELL
  3. EMA9 vs SMA21   — EMA9 > SMA21 for BUY
  4. MACD direction  — line > signal AND hist rising for BUY
  5. ADX > 20
  6. Volume > MA20
  7. Breakout        — HA_close > highest(HA_high, 10)[1] for BUY
  8. HA open filter  — HA_open > HA_open[1] for BUY
                       HA_open < HA_close[1] for SELL

SL = 1.5×ATR,  TP = 1.5×1.2×ATR  (R:R 1:1.2)
"""
import logging
import numpy as np
from .base import BaseStrategy, Signal, SignalType

logger = logging.getLogger("macd_ema_strategy")

_ATR_PERIOD  = 14
_SL_MULTS    = [1.0, 1.5, 2.0, 2.5]
_LOOKFORWARD = 60


class _HAC:
    """Lightweight Heikin Ashi candle object for indicator helpers."""
    __slots__ = ("timestamp", "open", "high", "low", "close", "volume")
    def __init__(self, ts, o, h, l, c, v):
        self.timestamp = ts
        self.open = o; self.high = h; self.low = l; self.close = c; self.volume = v


class MACDEMAStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.hma_period    = self.params.get("hma_period",    15)
        self.ema_fast      = self.params.get("ema_fast",       9)
        self.sma_slow      = self.params.get("sma_slow",      21)
        self.macd_fast     = self.params.get("macd_fast",     12)
        self.macd_slow     = self.params.get("macd_slow",     26)
        self.macd_sig      = self.params.get("macd_signal",    9)
        self.adx_len       = self.params.get("adx_len",       14)
        self.adx_threshold = self.params.get("adx_threshold", 20)
        self.breakout_len  = self.params.get("breakout_len",  10)
        self.vol_len       = self.params.get("vol_len",       20)
        self.ema50_len     = self.params.get("ema50_len",     50)
        self.sl_atr_mult   = self.params.get("sl_atr_mult",  1.5)
        self.rr_ratio      = self.params.get("rr_ratio",     1.2)

    # ── Heikin Ashi conversion ─────────────────────────────────────

    @staticmethod
    def _heikin_ashi(candles: list):
        """
        Convert regular OHLCV candles to Heikin Ashi.
        Returns (ha_candles_list, ha_open_arr, ha_close_arr).
        """
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

    # ── Signal helper ──────────────────────────────────────────────

    def _signal_at(self, i: int,
                   ha_closes, ha_highs, ha_lows, volumes,
                   hma15, ema9, sma21, ml, sl_line, hist,
                   adx_a, atr_a, vol_ma,
                   ha_open, ha_close_arr,
                   ema50_1h: float = float("nan")) -> int:
        """
        Returns +1 (BUY), -1 (SELL), 0 (HOLD) at bar i.
        All arrays computed on Heikin Ashi candles.
        ema50_1h=nan → skip 1H trend filter.
        """
        if i < max(2, self.breakout_len + 2):
            return 0

        check = [hma15[i], ema9[i], sma21[i], ml[i], sl_line[i],
                 hist[i], hist[i-1], vol_ma[i], ha_open[i], ha_open[i-1]]
        if any(np.isnan(v) for v in check):
            return 0

        p         = float(ha_closes[i])
        hma_slope = float(hma15[i]) - float(hma15[i-1])
        e9_c      = float(ema9[i]);     s21_c = float(sma21[i])
        ml_c      = float(ml[i]);       sl_c  = float(sl_line[i])
        h_c       = float(hist[i]);     h_p   = float(hist[i-1])
        adx_v     = float(adx_a[i]) if not np.isnan(adx_a[i]) else 0.0
        vol_v     = float(volumes[i]);  vol_ma_v = float(vol_ma[i])
        ha_o_c    = float(ha_open[i]);  ha_o_p = float(ha_open[i-1])
        ha_cl_p   = float(ha_close_arr[i-1])

        macd_bull = ml_c > sl_c and h_c > h_p
        macd_bear = ml_c < sl_c and h_c < h_p
        adx_ok    = adx_v > self.adx_threshold
        vol_ok    = vol_v > vol_ma_v

        # Breakout on HA highs/lows: close > highest(ha_high, N)[1]
        bo_lo  = max(0, i - self.breakout_len)
        ha_hs  = ha_highs[bo_lo:i]
        ha_ls  = ha_lows[bo_lo:i]
        if len(ha_hs) == 0:
            return 0
        bo_buy  = p > float(np.max(ha_hs))
        bo_sell = p < float(np.min(ha_ls))

        # HA open filters
        ha_buy_filter  = ha_o_c > ha_o_p     # HA_open > HA_open[1]
        ha_sell_filter = ha_o_c < ha_cl_p    # HA_open < HA_close[1]

        if np.isnan(ema50_1h):
            trend_bull = trend_bear = True
        else:
            trend_bull = p > ema50_1h
            trend_bear = p < ema50_1h

        if (trend_bull and hma_slope > 0 and e9_c > s21_c
                and macd_bull and adx_ok and vol_ok and bo_buy and ha_buy_filter):
            return 1
        if (trend_bear and hma_slope < 0 and e9_c < s21_c
                and macd_bear and adx_ok and vol_ok and bo_sell and ha_sell_filter):
            return -1
        return 0

    # ── Pre-compute all arrays (on HA candles) ─────────────────────

    def _build_arrays(self, candles: list):
        ha_candles, ha_o, ha_c = self._heikin_ashi(candles)

        ha_closes = ha_c
        ha_highs  = np.array([c.high   for c in ha_candles], dtype=float)
        ha_lows   = np.array([c.low    for c in ha_candles], dtype=float)
        volumes   = np.array([c.volume for c in candles],     dtype=float)

        ha_cl_list = ha_closes.tolist()
        hma15   = np.array(self.hma(ha_cl_list, self.hma_period),  dtype=float)
        ema9    = np.array(self.ema(ha_cl_list, self.ema_fast),     dtype=float)
        sma21   = np.array(self.sma(ha_cl_list, self.sma_slow),     dtype=float)
        _ml, _sl, _hi = self.macd(ha_cl_list, self.macd_fast, self.macd_slow, self.macd_sig)
        ml      = np.array(_ml, dtype=float)
        sl_line = np.array(_sl, dtype=float)
        hist    = np.array(_hi, dtype=float)
        adx_a, _, _ = self.adx(ha_candles, self.adx_len)
        adx_a   = np.array(adx_a, dtype=float)
        atr_a   = np.array(self.atr(ha_candles, _ATR_PERIOD), dtype=float)
        vol_ma  = np.array(self.sma(volumes.tolist(), self.vol_len), dtype=float)

        return (ha_closes, ha_highs, ha_lows, volumes,
                hma15, ema9, sma21, ml, sl_line, hist,
                adx_a, atr_a, vol_ma, ha_o, ha_c)

    # ── Live analysis ──────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = self.macd_slow + self.macd_sig + self.vol_len + self.hma_period + 5
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        (ha_closes, ha_highs, ha_lows, volumes,
         hma15, ema9, sma21, ml, sl_line, hist,
         adx_a, atr_a, vol_ma, ha_o, ha_c) = self._build_arrays(candles)

        # 1H EMA50 on HA candles
        ema50_1h = float("nan")
        if mtf_candles and "1h" in mtf_candles:
            h1 = mtf_candles["1h"]
            if len(h1) >= self.ema50_len:
                _, _, h1_hac = self._heikin_ashi(h1)
                ema50_1h = float(self.ema(h1_hac.tolist(), self.ema50_len)[-1])

        n = len(candles) - 1
        direction = self._signal_at(
            n, ha_closes, ha_highs, ha_lows, volumes,
            hma15, ema9, sma21, ml, sl_line, hist,
            adx_a, atr_a, vol_ma, ha_o, ha_c,
            ema50_1h=ema50_1h,
        )

        p     = current_price
        atr_c = float(atr_a[n]) if not np.isnan(atr_a[n]) else 0.0
        adx_v = float(adx_a[n]) if not np.isnan(adx_a[n]) else 0.0
        vol_ratio = (float(volumes[n]) / float(vol_ma[n])
                     if not np.isnan(vol_ma[n]) and vol_ma[n] > 0 else 0.0)

        meta = {
            "ha_close":  round(float(ha_closes[n]), 4),
            "ha_open":   round(float(ha_o[n]),       4),
            "hma15":     round(float(hma15[n]),      4),
            "ema9":      round(float(ema9[n]),        4),
            "sma21":     round(float(sma21[n]),       4),
            "macd":      round(float(ml[n]),          5),
            "hist":      round(float(hist[n]),        5),
            "adx":       round(adx_v,                 1),
            "vol_ratio": round(vol_ratio,              2),
            "ema50_1h":  round(ema50_1h, 4) if not np.isnan(ema50_1h) else None,
            "atr":       round(atr_c,                  4),
        }

        if direction == 1:
            sl_p = round(p - self.sl_atr_mult * atr_c, 4)
            tp_p = round(p + self.sl_atr_mult * self.rr_ratio * atr_c, 4)
            conf = round(min(0.90, 0.60 + max(0, adx_v - self.adx_threshold) / 80), 2)
            return Signal(
                type=SignalType.BUY, symbol=self.symbol, price=p, amount=0.0,
                confidence=conf,
                reason=(f"[MACD/EMA] BUY | HA↑ ADX={adx_v:.0f} Vol×{vol_ratio:.1f}"
                        + (f" 1H>{ema50_1h:.4f}" if not np.isnan(ema50_1h) else "")),
                metadata={**meta, "stop_loss": sl_p, "take_profit": tp_p, "rr": self.rr_ratio},
            )

        if direction == -1:
            sl_p = round(p + self.sl_atr_mult * atr_c, 4)
            tp_p = round(p - self.sl_atr_mult * self.rr_ratio * atr_c, 4)
            conf = round(min(0.90, 0.60 + max(0, adx_v - self.adx_threshold) / 80), 2)
            return Signal(
                type=SignalType.SELL, symbol=self.symbol, price=p, amount=0.0,
                confidence=conf,
                reason=(f"[MACD/EMA] SELL | HA↓ ADX={adx_v:.0f} Vol×{vol_ratio:.1f}"
                        + (f" 1H<{ema50_1h:.4f}" if not np.isnan(ema50_1h) else "")),
                metadata={**meta, "stop_loss": sl_p, "take_profit": tp_p, "rr": self.rr_ratio},
            )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[MACD/EMA] HOLD | ADX={adx_v:.1f} Vol×{vol_ratio:.1f}",
            metadata=meta,
        )

    # ── Backtest ───────────────────────────────────────────────────

    async def backtest(self, candles: list) -> tuple[dict, tuple]:
        min_len = self.macd_slow + self.macd_sig + self.vol_len + self.hma_period + 20
        if len(candles) < min_len:
            return {}, None

        (ha_closes, ha_highs, ha_lows, volumes,
         hma15, ema9, sma21, ml, sl_line, hist,
         adx_a, atr_a, vol_ma, ha_o, ha_c) = self._build_arrays(candles)

        signal_bars: list[tuple[int, int, float]] = []
        prev_dir = 0

        for i in range(min_len, len(candles) - 1):
            d = self._signal_at(i, ha_closes, ha_highs, ha_lows, volumes,
                                hma15, ema9, sma21, ml, sl_line, hist,
                                adx_a, atr_a, vol_ma, ha_o, ha_c)
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
                entry = float(ha_closes[idx])
                sl_p  = entry - sl_m * atr_val if direction == 1 else entry + sl_m * atr_val
                tp_p  = entry + sl_m * rr * atr_val if direction == 1 else entry - sl_m * rr * atr_val
                outcome = 0
                for j in range(idx + 1, min(idx + _LOOKFORWARD, len(ha_closes))):
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
