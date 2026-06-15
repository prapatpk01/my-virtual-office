"""
UT Bot v2 Strategy — ATR Trailing Stop.

BUY:  close crosses above ATR trailing stop line
SELL: close crosses below ATR trailing stop line

Trailing stop logic (exact Pine Script UT Bot v2):
  price > tsl[1] and price[1] > tsl[1]  → trail up:   max(tsl[1], price - sl)
  price < tsl[1] and price[1] < tsl[1]  → trail down:  min(tsl[1], price + sl)
  price > tsl[1]                         → flip bull:   price - sl
  else                                   → flip bear:   price + sl

Default: mult=1.1, atr_period=14
SL = 1.0×ATR(14), R:R = 1:1.0
BUY WR: ~60% across 250/500/750 bars 1H (tuned for 1H, ~0.8 trades/day)
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType

_ATR_PERIOD  = 14
_SL_MULTS    = [1.0, 1.5, 2.0, 2.5]
_LOOKFORWARD = 60


class UTBotStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.ut_mult     = self.params.get("ut_mult",     1.1)
        self.ut_atr_len  = self.params.get("ut_atr_len",  14)
        self.sl_atr_mult = self.params.get("sl_atr_mult", 1.0)
        self.rr_ratio    = self.params.get("rr_ratio",    1.0)
        self._last_signal = 0

    # ── ATR Trailing Stop ──────────────────────────────────────────

    def _trailing_stop(self, closes: np.ndarray, atr_a: np.ndarray) -> np.ndarray:
        """UT Bot v2 trailing stop — exact Pine Script logic."""
        n   = len(closes)
        tsl = np.full(n, np.nan)
        for i in range(n):
            slval = float(atr_a[i]) * self.ut_mult if not np.isnan(atr_a[i]) else 0.0
            src   = float(closes[i])
            if i == 0 or np.isnan(tsl[i - 1]):
                tsl[i] = src + slval
                continue
            tsl_p = float(tsl[i - 1])
            src_p = float(closes[i - 1])
            if src > tsl_p and src_p > tsl_p:
                tsl[i] = max(tsl_p, src - slval)
            elif src < tsl_p and src_p < tsl_p:
                tsl[i] = min(tsl_p, src + slval)
            elif src > tsl_p:
                tsl[i] = src - slval
            else:
                tsl[i] = src + slval
        return tsl

    # ── Signal arrays ──────────────────────────────────────────────

    def _build_signals(self, candles: list):
        """Returns (buy_sig, sell_sig, tsl, atr14)."""
        ha_candles, _, ha_c = self._heikin_ashi(candles)

        n      = len(candles)
        closes = ha_c
        highs  = np.array([c.high for c in ha_candles], dtype=float)
        lows   = np.array([c.low  for c in ha_candles], dtype=float)

        ut_atr = np.array(self.atr(ha_candles, self.ut_atr_len), dtype=float)
        atr14  = np.array(self.atr(ha_candles, _ATR_PERIOD),     dtype=float)
        tsl    = self._trailing_stop(closes, ut_atr)

        buy_sig  = np.zeros(n, dtype=bool)
        sell_sig = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if np.isnan(tsl[i]) or np.isnan(tsl[i - 1]):
                continue
            # crossover(close, tsl) — close crosses above tsl
            buy_sig[i]  = closes[i - 1] < tsl[i - 1] and closes[i] > tsl[i]
            # crossover(tsl, close) — tsl crosses above close
            sell_sig[i] = closes[i - 1] > tsl[i - 1] and closes[i] < tsl[i]

        return buy_sig, sell_sig, tsl, atr14

    # ── Live analysis ──────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = self.ut_atr_len + _ATR_PERIOD + 5
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        buy_sig, sell_sig, tsl, atr14 = self._build_signals(candles)

        n     = len(candles) - 1
        p     = current_price
        atr   = float(atr14[n]) if not np.isnan(atr14[n]) else 0.0
        tsl_v = float(tsl[n])   if not np.isnan(tsl[n])   else p
        trend = "Bull" if p > tsl_v else "Bear"
        conf  = round(min(0.85, 0.60 + abs(p - tsl_v) / max(p, 1) * 20), 2)

        if buy_sig[n] and self._last_signal != 1:
            self._last_signal = 1
            sl_p = round(p - self.sl_atr_mult * atr, 4)
            tp_p = round(p + self.sl_atr_mult * self.rr_ratio * atr, 4)
            return Signal(
                type=SignalType.BUY, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=f"[UT Bot] cross↑ TSL={tsl_v:.4f}",
                metadata={
                    "tsl": round(tsl_v, 4), "atr": round(atr, 4),
                    "stop_loss": sl_p, "take_profit": tp_p, "rr": self.rr_ratio,
                },
            )

        if sell_sig[n] and self._last_signal != -1:
            self._last_signal = -1
            sl_p = round(p + self.sl_atr_mult * atr, 4)
            tp_p = round(p - self.sl_atr_mult * self.rr_ratio * atr, 4)
            return Signal(
                type=SignalType.SELL, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=f"[UT Bot] cross↓ TSL={tsl_v:.4f}",
                metadata={
                    "tsl": round(tsl_v, 4), "atr": round(atr, 4),
                    "stop_loss": sl_p, "take_profit": tp_p, "rr": self.rr_ratio,
                },
            )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[UT Bot] {trend} | TSL={tsl_v:.4f}",
            metadata={"tsl": round(tsl_v, 4), "trend": trend},
        )

    # ── Backtest ───────────────────────────────────────────────────

    async def backtest(self, candles: list) -> tuple[dict, tuple]:
        min_len = self.ut_atr_len + _ATR_PERIOD + 20
        if len(candles) < min_len:
            return {}, None

        cls = np.array([c.close for c in candles], dtype=float)
        hig = np.array([c.high  for c in candles], dtype=float)
        low = np.array([c.low   for c in candles], dtype=float)

        buy_sig, sell_sig, _, atr14 = self._build_signals(candles)

        signal_bars = []
        prev_dir = 0
        for i in range(min_len, len(candles) - 1):
            if np.isnan(atr14[i]):
                continue
            if buy_sig[i] and prev_dir != 1:
                signal_bars.append((i, 1, float(atr14[i])))
                prev_dir = 1
            elif sell_sig[i] and prev_dir != -1:
                signal_bars.append((i, -1, float(atr14[i])))
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
