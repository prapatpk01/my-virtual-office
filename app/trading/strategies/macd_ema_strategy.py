"""
MACD + EMA Strategy — Institutional Momentum (SJ rewrite).

All conditions must pass (AND logic):
  1. 1H EMA50 trend  — close > EMA50_1h for BUY  (skip if 1H data unavailable)
  2. HMA15 slope     — rising for BUY, falling for SELL
  3. EMA9 vs SMA21   — EMA9 > SMA21 for BUY
  4. MACD direction  — line > signal AND hist rising for BUY
  5. ADX > 20        — confirms trending market
  6. Volume > MA20   — confirms participation
  7. Breakout        — close > highest(high, 10)[1] for BUY

SL = 1.5×ATR,  TP = 1.5×1.2×ATR  (R:R 1:1.2)
"""
import logging
import numpy as np
from .base import BaseStrategy, Signal, SignalType

logger = logging.getLogger("macd_ema_strategy")

_ATR_PERIOD  = 14
_SL_MULTS    = [1.0, 1.5, 2.0, 2.5]
_LOOKFORWARD = 60


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

    # ── Signal helper (called by analyze + backtest + backtest_full) ──

    def _signal_at(self, i: int,
                   closes, highs, lows, volumes,
                   hma15, ema9, sma21, ml, sl_line, hist,
                   adx_a, atr_a, vol_ma,
                   ema50_1h: float = float("nan")) -> int:
        """
        Returns +1 (BUY), -1 (SELL), 0 (HOLD) at bar i.
        ema50_1h=nan → skip 1H trend filter.
        """
        if i < max(2, self.breakout_len + 2):
            return 0

        check = [hma15[i], ema9[i], sma21[i], ml[i], sl_line[i],
                 hist[i], hist[i-1], vol_ma[i]]
        if any(np.isnan(v) for v in check):
            return 0

        p         = float(closes[i])
        hma_slope = float(hma15[i]) - float(hma15[i-1])
        e9_c      = float(ema9[i]);     s21_c = float(sma21[i])
        ml_c      = float(ml[i]);       sl_c  = float(sl_line[i])
        h_c       = float(hist[i]);     h_p   = float(hist[i-1])
        adx_v     = float(adx_a[i]) if not np.isnan(adx_a[i]) else 0.0
        vol_v     = float(volumes[i]);  vol_ma_v = float(vol_ma[i])

        macd_bull = ml_c > sl_c and h_c > h_p
        macd_bear = ml_c < sl_c and h_c < h_p
        adx_ok    = adx_v > self.adx_threshold
        vol_ok    = vol_v > vol_ma_v

        # ta.highest(high, N)[1] = max(high[i-N ... i-1])
        bo_lo  = max(0, i - self.breakout_len)
        hs     = highs[bo_lo:i]
        ls     = lows[bo_lo:i]
        if len(hs) == 0:
            return 0
        bo_buy  = p > float(np.max(hs))
        bo_sell = p < float(np.min(ls))

        if np.isnan(ema50_1h):
            trend_bull = trend_bear = True
        else:
            trend_bull = p > ema50_1h
            trend_bear = p < ema50_1h

        if (trend_bull and hma_slope > 0 and e9_c > s21_c
                and macd_bull and adx_ok and vol_ok and bo_buy):
            return 1
        if (trend_bear and hma_slope < 0 and e9_c < s21_c
                and macd_bear and adx_ok and vol_ok and bo_sell):
            return -1
        return 0

    # ── Build indicator arrays once ────────────────────────────────

    def _build_arrays(self, candles: list):
        closes  = [c.close  for c in candles]
        highs   = np.array([c.high   for c in candles], dtype=float)
        lows    = np.array([c.low    for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        hma15   = np.array(self.hma(closes, self.hma_period),  dtype=float)
        ema9    = np.array(self.ema(closes, self.ema_fast),     dtype=float)
        sma21   = np.array(self.sma(closes, self.sma_slow),     dtype=float)
        _ml, _sl, _hi = self.macd(closes, self.macd_fast, self.macd_slow, self.macd_sig)
        ml      = np.array(_ml,  dtype=float)
        sl_line = np.array(_sl,  dtype=float)
        hist    = np.array(_hi,  dtype=float)
        adx_a, _, _ = self.adx(candles, self.adx_len)
        adx_a   = np.array(adx_a, dtype=float)
        atr_a   = np.array(self.atr(candles, _ATR_PERIOD), dtype=float)
        vol_ma  = np.array(self.sma(volumes.tolist(), self.vol_len), dtype=float)
        closes_n = np.array(closes, dtype=float)

        return closes_n, highs, lows, volumes, hma15, ema9, sma21, ml, sl_line, hist, adx_a, atr_a, vol_ma

    # ── Live analysis ──────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = self.macd_slow + self.macd_sig + self.vol_len + self.hma_period + 5
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        (closes_n, highs, lows, volumes,
         hma15, ema9, sma21, ml, sl_line, hist,
         adx_a, atr_a, vol_ma) = self._build_arrays(candles)

        # 1H EMA50
        ema50_1h = float("nan")
        if mtf_candles and "1h" in mtf_candles:
            h1c = [c.close for c in mtf_candles["1h"]]
            if len(h1c) >= self.ema50_len:
                ema50_1h = float(self.ema(h1c, self.ema50_len)[-1])

        n = len(candles) - 1
        direction = self._signal_at(
            n, closes_n, highs, lows, volumes,
            hma15, ema9, sma21, ml, sl_line, hist,
            adx_a, atr_a, vol_ma, ema50_1h=ema50_1h,
        )

        p     = current_price
        atr_c = float(atr_a[n]) if not np.isnan(atr_a[n]) else 0.0
        adx_v = float(adx_a[n]) if not np.isnan(adx_a[n]) else 0.0
        vol_ratio = (float(volumes[n]) / float(vol_ma[n])
                     if not np.isnan(vol_ma[n]) and vol_ma[n] > 0 else 0.0)

        meta = {
            "hma15":     round(float(hma15[n]), 4),
            "ema9":      round(float(ema9[n]),  4),
            "sma21":     round(float(sma21[n]), 4),
            "macd":      round(float(ml[n]),    5),
            "hist":      round(float(hist[n]),  5),
            "adx":       round(adx_v,            1),
            "vol_ratio": round(vol_ratio,         2),
            "ema50_1h":  round(ema50_1h, 4) if not np.isnan(ema50_1h) else None,
            "atr":       round(atr_c,             4),
        }

        if direction == 1:
            sl_p = round(p - self.sl_atr_mult * atr_c, 4)
            tp_p = round(p + self.sl_atr_mult * self.rr_ratio * atr_c, 4)
            conf = round(min(0.90, 0.60 + max(0, adx_v - self.adx_threshold) / 80), 2)
            return Signal(
                type=SignalType.BUY, symbol=self.symbol, price=p, amount=0.0,
                confidence=conf,
                reason=(f"[MACD/EMA] BUY | ADX={adx_v:.0f} Vol×{vol_ratio:.1f}"
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
                reason=(f"[MACD/EMA] SELL | ADX={adx_v:.0f} Vol×{vol_ratio:.1f}"
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

        (closes_n, highs, lows, volumes,
         hma15, ema9, sma21, ml, sl_line, hist,
         adx_a, atr_a, vol_ma) = self._build_arrays(candles)

        signal_bars: list[tuple[int, int, float]] = []
        prev_dir = 0

        for i in range(min_len, len(candles) - 1):
            d = self._signal_at(i, closes_n, highs, lows, volumes,
                                hma15, ema9, sma21, ml, sl_line, hist,
                                adx_a, atr_a, vol_ma)
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
                entry = float(closes_n[idx])
                sl_p  = entry - sl_m * atr_val if direction == 1 else entry + sl_m * atr_val
                tp_p  = entry + sl_m * rr * atr_val if direction == 1 else entry - sl_m * rr * atr_val
                outcome = 0
                for j in range(idx + 1, min(idx + _LOOKFORWARD, len(candles))):
                    if direction == 1:
                        if lows[j]  <= sl_p: outcome = -1; break
                        if highs[j] >= tp_p: outcome =  1; break
                    else:
                        if highs[j] >= sl_p: outcome = -1; break
                        if lows[j]  <= tp_p: outcome =  1; break
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
