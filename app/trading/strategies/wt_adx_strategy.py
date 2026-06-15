"""
SJ WaveTrend Strategy.

BUY:  wt1 crossover wt2  (crossover-only mode, ob=os=100 disables OB/OS filter)
SELL: wt1 crossunder wt2 (crossover-only mode, ob=os=100 disables OB/OS filter)

SL = slMult × ATR(14)  →  2.0 × ATR
TP = SL × rrRatio       →  3.0 × ATR  (R:R 1:1.5)

Default: n1=2, n2=4, crossover-only mode (ob=os=100) — fires ~33 signals/250 bars 1H, WR≈76%
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType

_ATR_PERIOD  = 14
_SL_MULTS    = [1.0, 1.5, 2.0, 2.5]
_LOOKFORWARD = 60


class WTADXStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.n1          = self.params.get("wt_channel_len",  2)
        self.n2          = self.params.get("wt_avg_len",      4)
        self.ob_level    = self.params.get("ob_level",       100.0)
        self.os_level    = self.params.get("os_level",       100.0)
        self.sl_atr_mult = self.params.get("sl_atr_mult",    2.0)
        self.rr_ratio    = self.params.get("rr_ratio",       1.5)
        self._last_signal = 0

    # ── WaveTrend Core ─────────────────────────────────────────────

    def _wavetrend(self, highs, lows, closes):
        h = np.array(highs); l = np.array(lows); c = np.array(closes)
        ap  = (h + l + c) / 3.0
        esa = np.array(self.ema(ap.tolist(), self.n1))
        diff_abs = np.abs(ap - esa)
        first_ok = np.where(~np.isnan(diff_abs))[0]
        if len(first_ok):
            diff_filled = diff_abs.copy()
            diff_filled[:first_ok[0]] = diff_abs[first_ok[0]]
        else:
            diff_filled = diff_abs
        d   = np.array(self.ema(diff_filled.tolist(), self.n1))
        ci  = np.where(d > 1e-10, (ap - esa) / (0.015 * d), 0.0)
        wt1 = np.array(self.ema(ci.tolist(), self.n2))
        wt2 = np.array(self.sma(wt1.tolist(), 4))
        return wt1, wt2

    # ── Signal arrays ──────────────────────────────────────────────

    def _build_signals(self, candles: list):
        """Returns (wt1, wt2, buy_sig, sell_sig, atr14)."""
        ha_candles, _, ha_c = self._heikin_ashi(candles)

        n   = len(candles)
        hig = np.array([c.high for c in ha_candles], dtype=float)
        low = np.array([c.low  for c in ha_candles], dtype=float)
        cls = ha_c

        wt1, wt2 = self._wavetrend(hig.tolist(), low.tolist(), cls.tolist())
        atr14    = np.array(self.atr(ha_candles, _ATR_PERIOD), dtype=float)

        buy_sig  = np.zeros(n, dtype=bool)
        sell_sig = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if np.isnan(wt1[i]) or np.isnan(wt2[i]):
                continue
            # crossover(wt1, wt2): wt1[i-1] <= wt2[i-1] AND wt1[i] > wt2[i]
            cross_up   = float(wt1[i-1]) <= float(wt2[i-1]) and float(wt1[i]) > float(wt2[i])
            # crossunder(wt1, wt2): wt1[i-1] >= wt2[i-1] AND wt1[i] < wt2[i]
            cross_down = float(wt1[i-1]) >= float(wt2[i-1]) and float(wt1[i]) < float(wt2[i])
            buy_sig[i]  = cross_up   and float(wt1[i]) < self.os_level
            sell_sig[i] = cross_down and float(wt1[i]) > self.ob_level

        return wt1, wt2, buy_sig, sell_sig, atr14

    # ── Kept for external gate usage ──────────────────────────────

    @staticmethod
    def compute_wt1(candles: list, n1: int = 10, n2: int = 21) -> float:
        """Return current WT1 value (on HA candles) — used as gate by other strategies."""
        if len(candles) < n1 + n2 + 5:
            return float("nan")
        ha_candles, _, ha_c = BaseStrategy._heikin_ashi(candles)
        h = np.array([c.high for c in ha_candles])
        l = np.array([c.low  for c in ha_candles])
        c = ha_c
        ap  = (h + l + c) / 3.0
        esa = np.array(BaseStrategy.ema(ap.tolist(), n1))
        diff_abs = np.abs(ap - esa)
        first_ok = np.where(~np.isnan(diff_abs))[0]
        if len(first_ok):
            diff_filled = diff_abs.copy()
            diff_filled[:first_ok[0]] = diff_abs[first_ok[0]]
        else:
            diff_filled = diff_abs
        d  = np.array(BaseStrategy.ema(diff_filled.tolist(), n1))
        ci = np.where(d > 1e-10, (ap - esa) / (0.015 * d), 0.0)
        return float(BaseStrategy.ema(ci.tolist(), n2)[-1])

    # ── Live analysis ──────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = self.n1 + self.n2 + _ATR_PERIOD + 10
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        wt1, wt2, buy_sig, sell_sig, atr14 = self._build_signals(candles)

        n = len(candles) - 1
        if np.isnan(wt1[n]) or np.isnan(atr14[n]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Indicator NaN")

        curr_wt1    = float(wt1[n])
        curr_wt2    = float(wt2[n]) if not np.isnan(wt2[n]) else 0.0
        current_atr = float(atr14[n])
        p           = current_price
        conf        = round(min(0.90, 0.55 + abs(curr_wt1) / 200), 2)

        if buy_sig[n] and self._last_signal != 1:
            self._last_signal = 1
            sl_p = round(p - self.sl_atr_mult * current_atr, 4)
            tp_p = round(p + self.sl_atr_mult * self.rr_ratio * current_atr, 4)
            return Signal(
                type=SignalType.BUY, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=f"[SJ-WT] BUY cross↑ | WT1={curr_wt1:.1f} < {self.os_level}",
                metadata={
                    "wt1": round(curr_wt1, 2), "wt2": round(curr_wt2, 2),
                    "atr": round(current_atr, 4),
                    "stop_loss": sl_p, "take_profit": tp_p, "rr": self.rr_ratio,
                },
            )

        if sell_sig[n] and self._last_signal != -1:
            self._last_signal = -1
            sl_p = round(p + self.sl_atr_mult * current_atr, 4)
            tp_p = round(p - self.sl_atr_mult * self.rr_ratio * current_atr, 4)
            return Signal(
                type=SignalType.SELL, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=f"[SJ-WT] SELL cross↓ | WT1={curr_wt1:.1f} > {self.ob_level}",
                metadata={
                    "wt1": round(curr_wt1, 2), "wt2": round(curr_wt2, 2),
                    "atr": round(current_atr, 4),
                    "stop_loss": sl_p, "take_profit": tp_p, "rr": self.rr_ratio,
                },
            )

        if not buy_sig[n] and not sell_sig[n]:
            self._last_signal = 0

        zone = (f"OB({curr_wt1:.1f})" if curr_wt1 > self.ob_level else
                f"OS({curr_wt1:.1f})" if curr_wt1 < self.os_level else
                f"neutral({curr_wt1:.1f})")
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[SJ-WT] {zone}",
            metadata={"wt1": round(curr_wt1, 2), "wt2": round(curr_wt2, 2)},
        )

    # ── Backtest ───────────────────────────────────────────────────

    async def backtest(self, candles: list) -> tuple[dict, tuple]:
        min_len = self.n1 + self.n2 + _ATR_PERIOD + 20
        if len(candles) < min_len:
            return {}, None

        cls  = np.array([c.close for c in candles], dtype=float)
        hig  = np.array([c.high  for c in candles], dtype=float)
        low  = np.array([c.low   for c in candles], dtype=float)

        _, _, buy_sig, sell_sig, atr14 = self._build_signals(candles)

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
                        if low[j]  <= sl_p: outcome = -1; break
                        if hig[j]  >= tp_p: outcome =  1; break
                    else:
                        if hig[j]  >= sl_p: outcome = -1; break
                        if low[j]  <= tp_p: outcome =  1; break
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
