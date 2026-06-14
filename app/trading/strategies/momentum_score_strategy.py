"""
Momentum Score Strategy — RSI Divergence + 4-Phase Market Cycle.

Phase engine:
  Phase 1: RSI < 40 OR Bullish Divergence  (accumulation)
  Phase 2: was Phase 1 AND RSI crosses above EMA9  (uptrend start)
  Phase 3: RSI > 60 OR Bearish Divergence  (distribution)
  Phase 4: was Phase 3 AND RSI crosses below EMA9  (downtrend start)

Signals:
  BUY:  Phase 1 → 2 transition
  SELL: Phase 3 → 4 transition

Divergence (pivot of WMA(RSI,3), lookback=5):
  Bull: price low < pivot-bar price low  AND  RSI > pivot-bar RSI
  Bear: price high > pivot-bar price high AND  RSI < pivot-bar RSI

SL = 1.5×ATR(14), R:R = 1:1.5
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType

_ATR_PERIOD  = 14
_SL_MULTS    = [1.0, 1.5, 2.0, 2.5]
_LOOKFORWARD = 60


class MomentumScoreStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.rsi_len     = self.params.get("rsi_len",      14)
        self.pivot_len   = self.params.get("pivot_len",     5)
        self.rsi_smooth  = self.params.get("rsi_smooth",    3)
        self.ema_len     = self.params.get("ema_len",        9)
        self.sl_atr_mult = self.params.get("sl_atr_mult",  1.5)
        self.rr_ratio    = self.params.get("rr_ratio",     1.5)

    # ── Indicators ─────────────────────────────────────────────────

    @staticmethod
    def _rsi(closes: np.ndarray, period: int) -> np.ndarray:
        """Wilder's RSI."""
        n = len(closes)
        result = np.full(n, np.nan)
        if n < period + 1:
            return result
        deltas = np.diff(closes)
        gains  = np.where(deltas > 0,  deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_g  = float(np.mean(gains[:period]))
        avg_l  = float(np.mean(losses[:period]))
        result[period] = 100.0 if avg_l < 1e-10 else 100 - 100 / (1 + avg_g / avg_l)
        for i in range(period + 1, n):
            avg_g = (avg_g * (period - 1) + gains[i - 1]) / period
            avg_l = (avg_l * (period - 1) + losses[i - 1]) / period
            result[i] = 100.0 if avg_l < 1e-10 else 100 - 100 / (1 + avg_g / avg_l)
        return result

    @staticmethod
    def _wma(arr: np.ndarray, period: int) -> np.ndarray:
        """Weighted Moving Average (linear weights 1..period)."""
        n       = len(arr)
        result  = np.full(n, np.nan)
        weights = np.arange(1, period + 1, dtype=float)
        wsum    = weights.sum()
        for i in range(period - 1, n):
            w = arr[i - period + 1 : i + 1]
            if not np.any(np.isnan(w)):
                result[i] = float(np.dot(w, weights) / wsum)
        return result

    # ── Core signal engine ─────────────────────────────────────────

    def _build_signals(self, candles: list):
        """
        Compute all signal arrays over the full candle list.
        Returns (buy_sig, sell_sig, phase, rsi_src, atr_a).
        """
        n      = len(candles)
        closes = np.array([c.close for c in candles], dtype=float)
        highs  = np.array([c.high  for c in candles], dtype=float)
        lows   = np.array([c.low   for c in candles], dtype=float)

        rsi_raw = self._rsi(closes, self.rsi_len)
        rsi_src = self._wma(rsi_raw, self.rsi_smooth)
        rsi_ema = np.array(self.ema(rsi_src.tolist(), self.ema_len), dtype=float)
        atr_a   = np.array(self.atr(candles, _ATR_PERIOD), dtype=float)

        pl = self.pivot_len

        # Pivot detection on rsiSrc (Pine: pivothigh/pivotlow of rsiSrc, len=5)
        # At bar i, pivot bar = i-pl (confirmed by pl right bars)
        # Matches Pine order: update last values THEN check divergence
        last_price_hi = np.nan
        last_rsi_hi   = np.nan
        last_price_lo = np.nan
        last_rsi_lo   = np.nan

        bull_div = np.zeros(n, dtype=bool)
        bear_div = np.zeros(n, dtype=bool)

        for i in range(2 * pl, n):
            j = i - pl
            if j < pl:
                continue
            w = rsi_src[j - pl : j + pl + 1]
            if np.any(np.isnan(w)):
                continue

            # Pivot low of rsiSrc → potential bull divergence
            if float(rsi_src[j]) == float(np.min(w)):
                last_price_lo = float(lows[j])
                last_rsi_lo   = float(rsi_src[j])
                if (not np.isnan(rsi_src[i]) and
                        float(lows[i])    < last_price_lo and
                        float(rsi_src[i]) > last_rsi_lo):
                    bull_div[i] = True

            # Pivot high of rsiSrc → potential bear divergence
            if float(rsi_src[j]) == float(np.max(w)):
                last_price_hi = float(highs[j])
                last_rsi_hi   = float(rsi_src[j])
                if (not np.isnan(rsi_src[i]) and
                        float(highs[i])   > last_price_hi and
                        float(rsi_src[i]) < last_rsi_hi):
                    bear_div[i] = True

        # 4-Phase state machine — exact Pine Script if/else-if cascade
        phase    = np.ones(n, dtype=int)
        buy_sig  = np.zeros(n, dtype=bool)
        sell_sig = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if np.isnan(rsi_src[i]) or np.isnan(rsi_ema[i]):
                phase[i] = phase[i - 1]
                continue

            prev   = int(phase[i - 1])
            rsi_v  = float(rsi_src[i])
            rsi_e  = float(rsi_ema[i])
            rsi_vp = float(rsi_src[i - 1]) if not np.isnan(rsi_src[i - 1]) else rsi_v
            rsi_ep = float(rsi_ema[i - 1]) if not np.isnan(rsi_ema[i - 1]) else rsi_e

            cross_up   = rsi_vp <= rsi_ep and rsi_v > rsi_e
            cross_down = rsi_vp >= rsi_ep and rsi_v < rsi_e

            if rsi_v < 40 or bull_div[i]:
                phase[i] = 1
            elif prev == 1 and cross_up:
                phase[i] = 2
            elif rsi_v > 60 or bear_div[i]:
                phase[i] = 3
            elif prev == 3 and cross_down:
                phase[i] = 4
            else:
                phase[i] = prev

            buy_sig[i]  = phase[i] == 2 and phase[i - 1] == 1
            sell_sig[i] = phase[i] == 4 and phase[i - 1] == 3

        return buy_sig, sell_sig, phase, rsi_src, atr_a

    # ── Live analysis ──────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = self.rsi_len + self.rsi_smooth + 2 * self.pivot_len + self.ema_len + 5
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        buy_sig, sell_sig, phase, rsi_src, atr_a = self._build_signals(candles)

        n   = len(candles) - 1
        p   = current_price
        atr = float(atr_a[n])   if not np.isnan(atr_a[n])   else 0.0
        rsi = float(rsi_src[n]) if not np.isnan(rsi_src[n]) else 50.0
        ph  = int(phase[n])
        conf = round(min(0.85, 0.60 + abs(rsi - 50) / 100), 2)

        if buy_sig[n]:
            sl_p = round(p - self.sl_atr_mult * atr, 4)
            tp_p = round(p + self.sl_atr_mult * self.rr_ratio * atr, 4)
            return Signal(
                type=SignalType.BUY, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=f"[Momentum] Phase 1→2 | RSI={rsi:.1f}",
                metadata={
                    "phase": ph, "rsi": round(rsi, 1),
                    "stop_loss": sl_p, "take_profit": tp_p, "rr": self.rr_ratio,
                },
            )

        if sell_sig[n]:
            sl_p = round(p + self.sl_atr_mult * atr, 4)
            tp_p = round(p - self.sl_atr_mult * self.rr_ratio * atr, 4)
            return Signal(
                type=SignalType.SELL, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=f"[Momentum] Phase 3→4 | RSI={rsi:.1f}",
                metadata={
                    "phase": ph, "rsi": round(rsi, 1),
                    "stop_loss": sl_p, "take_profit": tp_p, "rr": self.rr_ratio,
                },
            )

        _NAMES = {1: "Accumulation", 2: "Uptrend", 3: "Distribution", 4: "Downtrend"}
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[Momentum] Phase {ph} ({_NAMES.get(ph,'?')}) | RSI={rsi:.1f}",
            metadata={"phase": ph, "rsi": round(rsi, 1)},
        )

    # ── Backtest ───────────────────────────────────────────────────

    async def backtest(self, candles: list) -> tuple[dict, tuple]:
        min_len = self.rsi_len + self.rsi_smooth + 2 * self.pivot_len + self.ema_len + 20
        if len(candles) < min_len:
            return {}, None

        cls = np.array([c.close for c in candles], dtype=float)
        hig = np.array([c.high  for c in candles], dtype=float)
        low = np.array([c.low   for c in candles], dtype=float)

        buy_sig, sell_sig, _, _, atr_a = self._build_signals(candles)

        signal_bars = []
        prev_dir = 0
        for i in range(min_len, len(candles) - 1):
            if np.isnan(atr_a[i]):
                continue
            if buy_sig[i] and prev_dir != 1:
                signal_bars.append((i, 1, float(atr_a[i])))
                prev_dir = 1
            elif sell_sig[i] and prev_dir != -1:
                signal_bars.append((i, -1, float(atr_a[i])))
                prev_dir = -1
            elif not buy_sig[i] and not sell_sig[i]:
                prev_dir = 0

        if not signal_bars:
            return {}, None

        best_score, best_config = -999.0, None
        stats: dict = {}

        for sl_m in _SL_MULTS:
            rr = self.rr_ratio
            wins = losses = 0
            total_r = 0.0
            for idx, direction, atr_val in signal_bars:
                if atr_val <= 0:
                    continue
                entry = float(cls[idx])
                sl_p  = entry - sl_m * atr_val if direction ==  1 else entry + sl_m * atr_val
                tp_p  = entry + sl_m * rr * atr_val if direction == 1 else entry - sl_m * rr * atr_val
                outcome = 0
                for j in range(idx + 1, min(idx + _LOOKFORWARD, len(cls))):
                    if direction == 1:
                        if low[j] <= sl_p: outcome = -1; break
                        if hig[j] >= tp_p: outcome =  1; break
                    else:
                        if hig[j] >= sl_p: outcome = -1; break
                        if low[j] <= tp_p: outcome =  1; break
                if outcome ==  1: wins   += 1; total_r += rr
                elif outcome == -1: losses += 1; total_r -= 1.0

            total = wins + losses
            wr  = wins / total if total else 0.0
            pf  = (wins * rr) / max(losses, 1)
            key = f"SL={sl_m}xATR  RR=1:{rr}"
            stats[key] = {
                "win_rate": round(wr * 100, 1), "profit_factor": round(pf, 2),
                "total_r":  round(total_r, 1),  "trades": total,
                "wins": wins,                    "losses": losses,
            }
            if total >= 5 and total_r > best_score:
                best_score  = total_r
                best_config = (sl_m, rr)

        return stats, best_config
