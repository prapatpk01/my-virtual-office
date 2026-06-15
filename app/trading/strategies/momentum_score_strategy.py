"""
Momentum Score Strategy — RSI × Heikin Ashi.

Signal fires when ALL 3 conditions are true:
  BUY:  HA bullish (HA_close > HA_open)  AND  RSI < 50  AND  RSI crosses above EMA9(RSI)
  SELL: HA bearish (HA_close < HA_open)  AND  RSI > 50  AND  RSI crosses below EMA9(RSI)

RSI cross is the trigger; HA direction + RSI zone confirm the trend.
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
        self.rsi_len     = self.params.get("rsi_len",     14)
        self.ema_len     = self.params.get("ema_len",      9)
        self.rsi_mid     = self.params.get("rsi_mid",     50)   # zone threshold
        self.sl_atr_mult = self.params.get("sl_atr_mult", 1.5)
        self.rr_ratio    = self.params.get("rr_ratio",    1.5)

    # ── Indicators ─────────────────────────────────────────────────

    @staticmethod
    def _ema_skipnan(arr: np.ndarray, period: int) -> np.ndarray:
        """EMA that skips leading NaN — needed when input (e.g. RSI) starts with NaN."""
        result = np.full(len(arr), np.nan)
        valid  = np.where(~np.isnan(arr))[0]
        if len(valid) == 0 or len(arr) - valid[0] < period:
            return result
        s = int(valid[0])
        k = 2.0 / (period + 1)
        result[s + period - 1] = float(np.mean(arr[s : s + period]))
        for i in range(s + period, len(arr)):
            result[i] = float(arr[i]) * k + result[i - 1] * (1 - k)
        return result

    @staticmethod
    def _rsi(closes: np.ndarray, period: int) -> np.ndarray:
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

    # ── Signal arrays ──────────────────────────────────────────────

    def _build_signals(self, candles: list):
        """Returns (buy_sig, sell_sig, rsi_a, rsi_ema_a, atr_a)."""
        ha_candles, ha_o, ha_c = self._heikin_ashi(candles)

        n = len(candles)
        atr_a   = np.array(self.atr(ha_candles, _ATR_PERIOD), dtype=float)
        rsi_a   = self._rsi(ha_c, self.rsi_len)
        rsi_ema = self._ema_skipnan(rsi_a, self.ema_len)

        buy_sig  = np.zeros(n, dtype=bool)
        sell_sig = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if np.isnan(rsi_a[i]) or np.isnan(rsi_a[i-1]):
                continue
            if np.isnan(rsi_ema[i]) or np.isnan(rsi_ema[i-1]):
                continue

            rsi_c  = float(rsi_a[i]);    rsi_p  = float(rsi_a[i-1])
            ema_c  = float(rsi_ema[i]);  ema_p  = float(rsi_ema[i-1])
            ha_o_v = float(ha_o[i]);     ha_c_v = float(ha_c[i])

            rsi_cross_up   = rsi_c > ema_c and rsi_p <= ema_p
            rsi_cross_down = rsi_c < ema_c and rsi_p >= ema_p
            ha_bull = ha_c_v > ha_o_v
            ha_bear = ha_c_v < ha_o_v

            buy_sig[i]  = ha_bull and rsi_c < self.rsi_mid and rsi_cross_up
            sell_sig[i] = ha_bear and rsi_c > self.rsi_mid and rsi_cross_down

        return buy_sig, sell_sig, rsi_a, rsi_ema, atr_a

    # ── Live analysis ──────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = self.rsi_len + self.ema_len + 5
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        buy_sig, sell_sig, rsi_a, rsi_ema, atr_a = self._build_signals(candles)

        n   = len(candles) - 1
        p   = current_price
        atr = float(atr_a[n]) if not np.isnan(atr_a[n]) else 0.0
        rsi = float(rsi_a[n]) if not np.isnan(rsi_a[n]) else 50.0
        conf = round(min(0.85, 0.60 + abs(rsi - 50) / 100), 2)

        meta = {
            "rsi":     round(rsi, 1),
            "rsi_ema": round(float(rsi_ema[n]), 1) if not np.isnan(rsi_ema[n]) else None,
            "atr":     round(atr, 4),
        }

        if buy_sig[n]:
            sl_p = round(p - self.sl_atr_mult * atr, 4)
            tp_p = round(p + self.sl_atr_mult * self.rr_ratio * atr, 4)
            return Signal(
                type=SignalType.BUY, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=f"[Momentum] BUY | HA↑ RSI={rsi:.1f}×↑EMA",
                metadata={**meta, "stop_loss": sl_p, "take_profit": tp_p, "rr": self.rr_ratio},
            )

        if sell_sig[n]:
            sl_p = round(p + self.sl_atr_mult * atr, 4)
            tp_p = round(p - self.sl_atr_mult * self.rr_ratio * atr, 4)
            return Signal(
                type=SignalType.SELL, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=f"[Momentum] SELL | HA↓ RSI={rsi:.1f}×↓EMA",
                metadata={**meta, "stop_loss": sl_p, "take_profit": tp_p, "rr": self.rr_ratio},
            )

        zone = "Bull" if rsi > 50 else "Bear"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[Momentum] HOLD | RSI={rsi:.1f} ({zone})",
            metadata=meta,
        )

    # ── Backtest ───────────────────────────────────────────────────

    async def backtest(self, candles: list) -> tuple[dict, tuple]:
        min_len = self.rsi_len + self.ema_len + 20
        if len(candles) < min_len:
            return {}, None

        ha_candles, ha_o, ha_c = self._heikin_ashi(candles)
        ha_highs = np.array([c.high for c in ha_candles], dtype=float)
        ha_lows  = np.array([c.low  for c in ha_candles], dtype=float)

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
